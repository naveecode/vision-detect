"""Train the crop CNN and report accuracy. This is not a detection metric."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from vision_detect.data.records import load_subset, records_for_split
from vision_detect.models.cnn import CropCNN
from vision_detect.utils.boxes import box_iou


def _crop(image: np.ndarray, box: list[float], size: int) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def _negative_crop(
    image: np.ndarray, boxes: list[list[float]], rng: np.random.Generator, size: int
):
    height, width = image.shape[:2]
    for _ in range(30):
        side = int(rng.integers(max(16, min(height, width) // 5), max(17, min(height, width) // 2)))
        if side >= width or side >= height:
            continue
        x1 = int(rng.integers(0, width - side))
        y1 = int(rng.integers(0, height - side))
        window = [x1, y1, x1 + side, y1 + side]
        if all(box_iou(window, box) < 0.05 for box in boxes):
            return _crop(image, window, size)
    return None


def _collect(records, image_dir, rng: np.random.Generator, size: int):
    images: list[np.ndarray] = []
    labels: list[int] = []
    for record in records:
        bgr = cv2.imread(str(image_dir / record["file_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_dir / record["file_name"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for box in record["boxes"]:
            positive = _crop(rgb, box, size)
            negative = _negative_crop(rgb, record["boxes"], rng, size)
            if positive is None or negative is None:
                continue
            images.extend([positive, negative])
            labels.extend([1, 0])
    stacked = np.stack(images).astype(np.float32) / 255.0
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    return tensor, torch.tensor(labels, dtype=torch.int64)


def _accuracy(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            logits = model(images[start : start + batch_size])
            correct += int((logits.argmax(dim=1) == labels[start : start + batch_size]).sum())
    return correct / max(len(labels), 1)


def train_cnn(args) -> dict:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    subset = load_subset(args.index)
    image_dir = args.images
    train_x, train_y = _collect(records_for_split(subset, "train"), image_dir, rng, args.size)
    val_x, val_y = _collect(records_for_split(subset, "val"), image_dir, rng, args.size)
    model = CropCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_y))
        total = 0.0
        steps = 0
        for start in range(0, len(train_y), args.batch_size):
            index = permutation[start : start + args.batch_size]
            logits = model(train_x[index])
            loss = loss_fn(logits, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        val_accuracy = _accuracy(model, val_x, val_y, args.batch_size)
        print(f"cnn epoch {epoch} loss={total / max(steps, 1):.4f} val_acc={val_accuracy:.4f}")
    # One-crop latency, batch size 1, after a warmup.
    model.eval()
    sample = val_x[:1]
    with torch.no_grad():
        model(sample)
        times = []
        for _ in range(30):
            started = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - started) * 1000.0)
    times.sort()
    result = {
        "task": "person-crop vs background-crop classification",
        "crop_size": args.size,
        "train_crops": int(len(train_y)),
        "val_crops": int(len(val_y)),
        "val_accuracy": _accuracy(model, val_x, val_y, args.batch_size),
        "positive_fraction_val": float(val_y.float().mean()),
        "latency_ms_median_batch1": times[len(times) // 2],
        "epochs": args.epochs,
        "note": "Accuracy is not detection mAP. The CNN does not localize.",
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result
