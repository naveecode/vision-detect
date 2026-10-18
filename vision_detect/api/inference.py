"""ONNX Runtime inference used by the FastAPI service."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from vision_detect.constants import DEFAULT_SCORE_THRESHOLD, INPUT_SIZE, MAX_UPLOAD_BYTES, NMS_IOU
from vision_detect.paths import metadata_path, model_path
from vision_detect.utils.boxes import postprocess


def preprocess_bgr(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Stretch a BGR image to the network input. Values are RGB in ``[0, 1]``."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor)


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if len(payload) == 0:
        raise ValueError("Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds 8 MB")
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


class OnnxPersonDetector:
    def __init__(self, model_file: Path | str, metadata_file: Path | str | None = None) -> None:
        self.model_file = Path(model_file)
        self.metadata: dict = {}
        if metadata_file and Path(metadata_file).exists():
            self.metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(
            str(self.model_file), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(self.metadata.get("input_size", INPUT_SIZE))
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> OnnxPersonDetector:
        return cls(model_path(), metadata_path())

    def version(self) -> dict:
        return {
            "model_name": self.metadata.get("model_name", "ssdlite320_mobilenet_v3_large"),
            "model_version": self.metadata.get("model_version", "unversioned"),
            "framework": "onnxruntime",
            "input_size": self.input_size,
            "classes": [self.metadata.get("class_name", "person")],
            "dataset": self.metadata.get("dataset", ""),
            "nms_iou": float(self.metadata.get("nms_iou", NMS_IOU)),
            "sha256": self.metadata.get("sha256", ""),
        }

    def predict(self, image_bgr: np.ndarray, score_threshold: float | None = None) -> dict:
        start = time.perf_counter()
        tensor = preprocess_bgr(image_bgr, self.input_size)
        with self._lock:
            boxes, scores = self.session.run(None, {self.input_name: tensor})
        threshold = (
            float(self.metadata.get("default_score_threshold", DEFAULT_SCORE_THRESHOLD))
            if score_threshold is None
            else float(score_threshold)
        )
        iou = float(self.metadata.get("nms_iou", NMS_IOU))
        height, width = image_bgr.shape[:2]
        kept_boxes, kept_scores = postprocess(
            boxes[0], scores[0], width, height, threshold, iou, self.input_size
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "boxes": [[round(float(value), 2) for value in box] for box in kept_boxes],
            "labels": ["person"] * len(kept_scores),
            "scores": [round(float(score), 4) for score in kept_scores],
            "image_size": {"width": int(width), "height": int(height)},
            "latency_ms": round(latency_ms, 2),
            "model_version": self.version()["model_version"],
        }
