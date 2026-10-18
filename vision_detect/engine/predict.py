"""Render sample predictions and copy redistributable photos for the browser demo."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vision_detect.api.inference import OnnxPersonDetector
from vision_detect.data.records import load_subset, records_for_split
from vision_detect.paths import REPO_ROOT
from vision_detect.viz.draw import draw_detections


def _resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 0.999:
        return image
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def render_samples(args) -> list[dict]:
    subset = load_subset(args.index)
    records = [
        record
        for record in records_for_split(subset, "val")
        if record.get("redistributable")
    ]
    if not records:
        raise RuntimeError("No redistributable validation images in the subset")
    detector = OnnxPersonDetector(args.model, args.metadata)
    docs_dir = Path(args.docs)
    web_dir = Path(args.web)
    docs_dir.mkdir(parents=True, exist_ok=True)
    web_dir.mkdir(parents=True, exist_ok=True)
    ranked: list[dict] = []
    for record in records:
        path = Path(args.images) / record["file_name"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        preview = _resize_max(image, args.max_side)
        result = detector.predict(preview, score_threshold=args.score_threshold)
        top = result["scores"][0] if result["scores"] else 0.0
        ranked.append({"top": top, "record": record, "preview": preview, "result": result})
    ranked.sort(key=lambda item: item["top"], reverse=True)
    chosen = ranked[: args.limit]
    manifest = []
    for item in chosen:
        record = item["record"]
        stem = Path(record["file_name"]).stem
        sample_name = f"{stem}.jpg"
        overlay_name = f"{stem}_pred.jpg"
        preview = item["preview"]
        cv2.imwrite(
            str(web_dir / sample_name), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        overlay = draw_detections(
            preview, item["result"]["boxes"], item["result"]["scores"], item["result"]["labels"]
        )
        cv2.imwrite(str(docs_dir / overlay_name), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        entry = {
            "file": sample_name,
            "overlay": overlay_name,
            "coco_id": record["id"],
            "license_name": record["license_name"],
            "license_url": record["license_url"],
            "flickr_url": record.get("flickr_url", ""),
            "coco_url": record["coco_url"],
            "score_threshold": args.score_threshold,
            "scores": item["result"]["scores"],
            "boxes": item["result"]["boxes"],
            "width": int(preview.shape[1]),
            "height": int(preview.shape[0]),
        }
        manifest.append(entry)
        print(f"sample {sample_name} top={item['top']:.3f} boxes={len(item['result']['boxes'])}")
    payload = json.dumps({"samples": manifest}, indent=2) + "\n"
    (web_dir / "manifest.json").write_text(payload, encoding="utf-8")
    (docs_dir / "manifest.json").write_text(payload, encoding="utf-8")
    return manifest


def default_docs() -> Path:
    return REPO_ROOT / "docs" / "predictions"


def default_web() -> Path:
    return REPO_ROOT / "web" / "samples"
