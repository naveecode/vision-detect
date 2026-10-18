"""COCO-style mAP via torchmetrics (pycocotools backend) and latency."""

from __future__ import annotations

import math
import time

import torch

from vision_detect.constants import NMS_IOU, PERSON_INDEX


def _metric_float(value) -> float | None:
    number = float(value)
    if math.isnan(number):
        return None
    return number


@torch.no_grad()
def evaluate_map(model, loader, device: torch.device) -> dict[str, float | None]:
    from torchmetrics.detection import MeanAveragePrecision

    was_training = model.training
    model.eval()
    model.score_thresh = 0.01
    model.nms_thresh = NMS_IOU
    model.detections_per_img = 100
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    for images, batch_targets in loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, batch_targets, strict=True):
            keep = output["labels"] == PERSON_INDEX
            count = int(keep.sum())
            predictions.append(
                {
                    "boxes": output["boxes"][keep].detach().cpu(),
                    "scores": output["scores"][keep].detach().cpu(),
                    "labels": torch.ones(count, dtype=torch.int64),
                }
            )
            gt_count = int(target["boxes"].shape[0])
            targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": torch.ones(gt_count, dtype=torch.int64),
                }
            )
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(predictions, targets)
    result = metric.compute()
    if was_training:
        model.train()
    return {
        "map": _metric_float(result["map"]),
        "map_50": _metric_float(result["map_50"]),
        "map_75": _metric_float(result["map_75"]),
        "mar_100": _metric_float(result["mar_100"]),
    }


@torch.no_grad()
def pytorch_latency_ms(model, dataset, device: torch.device, limit: int = 20) -> dict:
    """Median single-image latency. Includes the model's own resize and NMS."""
    model.eval()
    count = min(limit, len(dataset))
    warmup = dataset[0][0].to(device)
    model([warmup])
    times: list[float] = []
    for index in range(count):
        image = dataset[index][0].to(device)
        start = time.perf_counter()
        model([image])
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
        "images": count,
        "device": str(device),
        "batch_size": 1,
    }
