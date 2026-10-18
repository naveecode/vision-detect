from pathlib import Path

import numpy as np
import torch
from torch import nn

from vision_detect.api.inference import OnnxPersonDetector
from vision_detect.engine.export import export_onnx, max_abs_error
from vision_detect.paths import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH


def test_tiny_onnx_matches_pytorch(tmp_path: Path):
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(3, 2, kernel_size=3, padding=1), nn.ReLU())
    module.eval()
    example = torch.randn(1, 3, 16, 16)
    path = tmp_path / "tiny.onnx"
    export_onnx(module, example, path, ["y"])
    assert max_abs_error(module, path, example) < 1e-4


class _FixedBoxes(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Add a slice of the input so the graph keeps an input. The test image is
        # zeros, so the added shift is 0 and the box stays [0, 0, 160, 320].
        shift = images[:, 0, 0, 0]
        boxes = images.new_tensor([0.0, 0.0, 160.0, 320.0]).view(1, 1, 4)
        boxes = boxes + shift.view(-1, 1, 1)
        scores = images.new_tensor([0.9]).view(1, 1) + shift.view(-1, 1) * 0
        return boxes, scores


def test_onnx_detector_scales_boxes(tmp_path: Path):
    module = _FixedBoxes().eval()
    path = tmp_path / "fixed.onnx"
    export_onnx(module, torch.zeros(1, 3, 320, 320), path, ["boxes", "scores"])
    detector = OnnxPersonDetector(path)
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    result = detector.predict(image, score_threshold=0.4)
    assert result["labels"] == ["person"]
    assert result["scores"] == [0.9]
    assert result["boxes"] == [[0.0, 0.0, 320.0, 320.0]]
    assert result["image_size"] == {"width": 640, "height": 320}
    assert result["latency_ms"] >= 0


def test_shipped_onnx_is_small_and_runs():
    assert DEFAULT_MODEL_PATH.exists(), "export the detector before CI"
    assert DEFAULT_MODEL_PATH.stat().st_size < 25 * 1024 * 1024
    meta = DEFAULT_METADATA_PATH.read_text(encoding="utf-8")
    assert "ssdlite320_mobilenet_v3_large" in meta
    detector = OnnxPersonDetector(DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH)
    result = detector.predict(np.zeros((180, 240, 3), dtype=np.uint8), score_threshold=0.4)
    assert isinstance(result["boxes"], list)
    assert len(result["boxes"]) == len(result["scores"]) == len(result["labels"])
    assert result["model_version"]
