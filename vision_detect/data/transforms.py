"""Detection augmentations.

Geometric flip is OpenCV so the box math is exact. Photometric changes and a
second flip path go through Albumentations, which is the usual detection-aug
library and is itself built on OpenCV.
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np

_AUGMENT = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
        clip=True,
    ),
)


def opencv_brightness(image: np.ndarray, delta: int) -> np.ndarray:
    """Shift uint8 brightness by ``delta`` using OpenCV. ``delta`` may be negative."""
    return cv2.convertScaleAbs(image, alpha=1.0, beta=float(delta))


def horizontal_flip(
    image: np.ndarray, boxes: list[list[float]]
) -> tuple[np.ndarray, list[list[float]]]:
    """Flip an image left-right and mirror xyxy boxes."""
    flipped = cv2.flip(image, 1)
    width = float(image.shape[1])
    mirrored = [[width - x2, y1, width - x1, y2] for x1, y1, x2, y2 in boxes]
    return flipped, mirrored


def augment_detection(
    image: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    training: bool,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply training augmentations. Evaluation returns the inputs unchanged."""
    if not training:
        return image, boxes, labels
    generator = rng or np.random.default_rng()
    delta = int(generator.integers(-18, 19))
    adjusted = opencv_brightness(image, delta)
    if not boxes:
        return adjusted, boxes, labels
    transformed = _AUGMENT(image=adjusted, bboxes=boxes, labels=labels)
    new_image = transformed["image"]
    if new_image.dtype != np.uint8:
        new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    new_boxes = [[float(v) for v in box] for box in transformed["bboxes"]]
    new_labels = [int(label) for label in transformed["labels"]]
    return new_image, new_boxes, new_labels
