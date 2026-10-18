"""COCO JSON parsing for the person class. Pure Python so tests stay light."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_detect.constants import IMAGE_URL, PERSON_INDEX


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, width, height = [float(v) for v in bbox]
    return [x, y, x + width, y + height]


def sanitize_boxes(
    boxes: list[list[float]],
    labels: list[int],
    width: int,
    height: int,
    min_size: float = 1.0,
) -> tuple[list[list[float]], list[int]]:
    """Clip boxes to the image and drop boxes that collapse."""
    kept_boxes: list[list[float]] = []
    kept_labels: list[int] = []
    max_x = float(width)
    max_y = float(height)
    for box, label in zip(boxes, labels, strict=True):
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = min(max(x1, 0.0), max_x)
        x2 = min(max(x2, 0.0), max_x)
        y1 = min(max(y1, 0.0), max_y)
        y2 = min(max(y2, 0.0), max_y)
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        kept_boxes.append([x1, y1, x2, y2])
        kept_labels.append(int(label))
    return kept_boxes, kept_labels


def license_allows_redistribution(name: str) -> bool:
    """True for licenses that allow sharing a resized image and a boxed derivative.

    Non-commercial and no-derivatives Flickr licenses are excluded. The project
    does not redistribute those images; training downloads them locally.
    """
    lowered = name.lower().replace(" ", "")
    if not lowered:
        return False
    if "noncommercial" in lowered or "non-commercial" in lowered:
        return False
    if "noderiv" in lowered or "noderivs" in lowered:
        return False
    return True


def person_records_from_coco(coco: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Keep non-crowd person boxes and convert them to xyxy records."""
    licenses = {int(item["id"]): item for item in coco.get("licenses", [])}
    boxes_by_image: dict[int, list[list[float]]] = {}
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) != PERSON_INDEX:
            continue
        if int(ann.get("iscrowd", 0)) != 0:
            continue
        boxes_by_image.setdefault(int(ann["image_id"]), []).append(xywh_to_xyxy(ann["bbox"]))

    records: list[dict[str, Any]] = []
    for image in coco.get("images", []):
        raw_boxes = boxes_by_image.get(int(image["id"]))
        if not raw_boxes:
            continue
        boxes, labels = sanitize_boxes(
            raw_boxes, [PERSON_INDEX] * len(raw_boxes), int(image["width"]), int(image["height"])
        )
        if not boxes:
            continue
        license_id = image.get("license")
        license_info = licenses.get(int(license_id)) if license_id is not None else {}
        license_name = str((license_info or {}).get("name", ""))
        file_name = str(image["file_name"])
        records.append(
            {
                "id": int(image["id"]),
                "file_name": file_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "license": None if license_id is None else int(license_id),
                "license_name": license_name,
                "license_url": str((license_info or {}).get("url", "")),
                "redistributable": license_allows_redistribution(license_name),
                "coco_url": image.get("coco_url")
                or IMAGE_URL.format(split=split, file_name=file_name),
                "flickr_url": image.get("flickr_url", ""),
                "split": split,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return records


def load_subset(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "images" not in data:
        raise ValueError(f"{path} is not a vision-detect subset index")
    return data


def records_for_split(subset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [record for record in subset["images"] if record["split"] == split]
