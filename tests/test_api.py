import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision_detect.api.app import create_app
from vision_detect.api.inference import decode_image_bytes
from vision_detect.constants import MAX_UPLOAD_BYTES


class FakeDetector:
    def version(self) -> dict:
        return {
            "model_name": "fake",
            "model_version": "test",
            "framework": "fake",
            "input_size": 320,
            "classes": ["person"],
            "dataset": "fixture",
            "nms_iou": 0.55,
            "sha256": "abc",
        }

    def predict(self, image_bgr: np.ndarray, score_threshold: float | None = None) -> dict:
        return {
            "boxes": [[1.0, 2.0, 3.0, 4.0]],
            "labels": ["person"],
            "scores": [0.5 if score_threshold is None else score_threshold],
            "image_size": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
            "latency_ms": 1.25,
            "model_version": "test",
        }


def _jpeg_bytes() -> bytes:
    image = np.full((32, 40, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_health_without_model():
    client = TestClient(create_app(load_default=False))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    rejected = client.post("/detect", files={"file": ("a.jpg", _jpeg_bytes(), "image/jpeg")})
    assert rejected.status_code == 503


def test_detect_contract():
    client = TestClient(create_app(detector=FakeDetector(), load_default=False))
    health = client.get("/health")
    assert health.json()["status"] == "ok"
    version = client.get("/version")
    assert version.status_code == 200
    assert version.json()["model_version"] == "test"
    detected = client.post(
        "/detect?score_threshold=0.2",
        files={"file": ("street.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert detected.status_code == 200
    body = detected.json()
    assert body["boxes"] == [[1.0, 2.0, 3.0, 4.0]]
    assert body["labels"] == ["person"]
    assert body["scores"] == [0.2]
    assert body["latency_ms"] == 1.25
    assert body["image_size"] == {"width": 40, "height": 32}


def test_rejects_bad_and_large_uploads():
    client = TestClient(create_app(detector=FakeDetector(), load_default=False))
    bad = client.post("/detect", files={"file": ("a.jpg", b"not-an-image", "image/jpeg")})
    assert bad.status_code == 400
    out_of_range = client.post(
        "/detect?score_threshold=1.5",
        files={"file": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert out_of_range.status_code == 422
    with pytest.raises(ValueError):
        decode_image_bytes(b"\x00" * (MAX_UPLOAD_BYTES + 1))


def test_grayscale_decode():
    gray = np.full((12, 12), 90, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", gray)
    assert ok
    image = decode_image_bytes(encoded.tobytes())
    assert image.shape == (12, 12, 3)
