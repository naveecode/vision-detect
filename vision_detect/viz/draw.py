"""Draw detection boxes with OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

# BGR, deep green so overlays stay readable on street photos.
BOX_COLOR = (70, 120, 46)
TEXT_COLOR = (245, 245, 236)


def draw_detections(
    image_bgr: np.ndarray,
    boxes: list[list[float]] | np.ndarray,
    scores: list[float] | np.ndarray,
    labels: list[str],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for box, score, label in zip(boxes, scores, labels, strict=True):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        text = f"{label} {float(score):.2f}"
        origin = (x1, max(16, y1 - 6))
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOX_COLOR,
            1,
            cv2.LINE_AA,
        )
    return canvas
