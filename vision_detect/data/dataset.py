"""Torch dataset over the committed COCO person subset index."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vision_detect.constants import PERSON_INDEX
from vision_detect.data.records import load_subset, records_for_split, sanitize_boxes
from vision_detect.data.transforms import augment_detection


class CocoPersonSubset(Dataset):
    """Person boxes from ``coco_person_subset.json`` plus locally downloaded images."""

    def __init__(
        self,
        index_path: Path | str,
        image_dir: Path | str,
        split: str,
        training: bool = False,
        seed: int = 0,
    ) -> None:
        subset = load_subset(index_path)
        self.records = records_for_split(subset, split)
        if not self.records:
            raise ValueError(f"No records for split {split!r} in {index_path}")
        self.image_dir = Path(image_dir)
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        path = self.image_dir / record["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m vision_detect download` first."
            )
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = [list(map(float, box)) for box in record["boxes"]]
        labels = [int(label) for label in record.get("labels", [PERSON_INDEX] * len(boxes))]
        if self.training:
            rng = np.random.default_rng(self.seed + index * 1009 + len(boxes))
            augmented, aug_boxes, aug_labels = augment_detection(
                image, boxes, labels, training=True, rng=rng
            )
            aug_boxes, aug_labels = sanitize_boxes(
                aug_boxes, aug_labels, image.shape[1], image.shape[0]
            )
            if aug_boxes:
                image, boxes, labels = augmented, aug_boxes, aug_labels
        height, width = image.shape[:2]
        boxes, labels = sanitize_boxes(boxes, labels, width, height)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return tensor, target


def collate_detections(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
