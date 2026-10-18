"""Build a small, deterministic COCO 2017 person subset.

The annotations zip is CC BY 4.0. Images are not written into the subset file;
each record keeps the upstream license so sample images can be filtered later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from vision_detect.constants import ANNOTATIONS_MD5, ANNOTATIONS_URL, IMAGE_URL, PERSON_INDEX
from vision_detect.data.records import license_allows_redistribution, sanitize_boxes, xywh_to_xyxy
from vision_detect.paths import REPO_ROOT, SUBSET_PATH

DEFAULT_ZIP = REPO_ROOT / "data" / "annotations" / "annotations_trainval2017.zip"
DEFAULT_EXTRACT = REPO_ROOT / "data" / "annotations" / "raw"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_annotations(zip_path: Path = DEFAULT_ZIP) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists() or _md5(zip_path) != ANNOTATIONS_MD5:
        print(f"Downloading COCO annotations to {zip_path}")
        request = urllib.request.Request(ANNOTATIONS_URL, headers={"User-Agent": "vision-detect/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, zip_path.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        digest = _md5(zip_path)
        if digest != ANNOTATIONS_MD5:
            raise RuntimeError(f"Annotations MD5 {digest} != {ANNOTATIONS_MD5}")
    return zip_path


def extract_instances(zip_path: Path, extract_dir: Path = DEFAULT_EXTRACT) -> dict[str, Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "annotations/instances_train2017.json",
        "val": "annotations/instances_val2017.json",
    }
    paths: dict[str, Path] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for split, name in names.items():
            dest = extract_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                print(f"Extracting {name}")
                archive.extract(name, extract_dir)
            paths[split] = dest
    return paths


def _stream_person_boxes(path: Path) -> dict[int, list[list[float]]]:
    import ijson

    boxes: dict[int, list[list[float]]] = defaultdict(list)
    seen = 0
    with path.open("rb") as handle:
        for ann in ijson.items(handle, "annotations.item"):
            seen += 1
            if seen % 200000 == 0:
                print(f"  scanned {seen} annotations in {path.name}")
            if int(ann.get("category_id", -1)) != PERSON_INDEX or int(ann.get("iscrowd", 0)) != 0:
                continue
            boxes[int(ann["image_id"])].append(xywh_to_xyxy(ann["bbox"]))
    print(f"  {path.name}: {len(boxes)} images with person boxes")
    return boxes


def _stream_licenses(path: Path) -> dict[int, dict[str, Any]]:
    import ijson

    licenses: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for item in ijson.items(handle, "licenses.item"):
            licenses[int(item["id"])] = item
    return licenses


def _qualifies(width: int, height: int, boxes: list[list[float]]) -> bool:
    if not (1 <= len(boxes) <= 3):
        return False
    if width < 240 or height < 240:
        return False
    image_area = float(width * height)
    ratios = []
    for x1, y1, x2, y2 in boxes:
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 1 or box_h < 1:
            return False
        ratios.append((box_w * box_h) / image_area)
    if min(ratios) < 0.01 or max(ratios) < 0.08 or max(ratios) > 0.85:
        return False
    return True


def _candidates(path: Path, split: str) -> list[dict[str, Any]]:
    boxes_by_id = _stream_person_boxes(path)
    licenses = _stream_licenses(path)
    import ijson

    chosen: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for image in ijson.items(handle, "images.item"):
            image_id = int(image["id"])
            raw = boxes_by_id.get(image_id)
            if not raw:
                continue
            width, height = int(image["width"]), int(image["height"])
            boxes, labels = sanitize_boxes(raw, [PERSON_INDEX] * len(raw), width, height)
            if not _qualifies(width, height, boxes):
                continue
            license_id = image.get("license")
            info = licenses.get(int(license_id), {}) if license_id is not None else {}
            name = str(info.get("name", ""))
            file_name = str(image["file_name"])
            chosen.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "license": None if license_id is None else int(license_id),
                    "license_name": name,
                    "license_url": str(info.get("url", "")),
                    "redistributable": license_allows_redistribution(name),
                    "coco_url": IMAGE_URL.format(split=split, file_name=file_name),
                    "flickr_url": image.get("flickr_url", ""),
                    "split": split,
                    "boxes": boxes,
                    "labels": labels,
                }
            )
    chosen.sort(key=lambda record: record["id"])
    return chosen


def _take_val(candidates: list[dict[str, Any]], limit: int, min_redistributable: int) -> list[dict[str, Any]]:
    redistributable = [record for record in candidates if record["redistributable"]]
    picked = redistributable[:min_redistributable]
    picked_ids = {record["id"] for record in picked}
    for record in candidates:
        if record["id"] in picked_ids:
            continue
        picked.append(record)
        if len(picked) >= limit:
            break
    return picked[:limit]


def build_subset(
    train_limit: int = 160,
    val_limit: int = 40,
    min_redistributable_val: int = 8,
    zip_path: Path = DEFAULT_ZIP,
    output_path: Path = SUBSET_PATH,
) -> Path:
    archive = download_annotations(zip_path)
    extracted = extract_instances(archive)
    print("Selecting train images")
    train = _candidates(extracted["train"], "train")[:train_limit]
    print("Selecting val images")
    val_all = _candidates(extracted["val"], "val")
    val = _take_val(val_all, val_limit, min_redistributable_val)
    if len(train) < train_limit or len(val) < val_limit:
        raise RuntimeError(f"Not enough images (train={len(train)}, val={len(val)})")
    redistributable = sum(1 for record in val if record["redistributable"])
    document = {
        "info": {
            "source": "COCO 2017 person subset",
            "homepage": "https://cocodataset.org/",
            "citation": (
                "Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, "
                "Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. "
                "Microsoft COCO: Common Objects in Context. ECCV 2014."
            ),
            "annotations_license": "Creative Commons Attribution 4.0",
            "annotations_url": "https://cocodataset.org/#termsofuse",
            "image_terms": (
                "Images keep the per-record Flickr license. This file redistributes "
                "annotations only. Records with redistributable=true allow a resized "
                "copy and a detection overlay (no non-commercial, no no-derivatives)."
            ),
            "download_url": ANNOTATIONS_URL,
            "annotations_md5": ANNOTATIONS_MD5,
            "selection": {
                "category_id": PERSON_INDEX,
                "iscrowd": 0,
                "people_per_image": [1, 3],
                "min_box_area_ratio": 0.01,
                "min_largest_box_area_ratio": 0.08,
                "max_box_area_ratio": 0.85,
                "min_side": 240,
                "order": "ascending COCO image id",
                "train_images": len(train),
                "val_images": len(val),
                "val_reserves_redistributable": min_redistributable_val,
            },
        },
        "images": train + val,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {output_path} train={len(train)} val={len(val)} "
        f"redistributable_val={redistributable}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the COCO person subset index")
    parser.add_argument("--train-images", type=int, default=160)
    parser.add_argument("--val-images", type=int, default=40)
    parser.add_argument("--output", type=Path, default=SUBSET_PATH)
    args = parser.parse_args()
    build_subset(args.train_images, args.val_images, output_path=args.output)


if __name__ == "__main__":
    main()
