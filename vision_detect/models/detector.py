"""Torchvision detection builders.

SSDLite320 MobileNetV3 is the shipped model: it fine-tunes like any torchvision
detector and its head exports to ONNX without RoIAlign. Faster R-CNN MobileNetV3
uses the same training loop when a two-stage ablation is useful, and stays in
PyTorch because that graph does not run in onnxruntime-web.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    name = arch.lower()
    if name == "ssdlite":
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        if pretrained:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            return ssdlite320_mobilenet_v3_large(weights=weights)
        return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    if name in {"fasterrcnn", "fasterrcnn_mobilenet"}:
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        if pretrained:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            return fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=91
        )
    raise ValueError(f"Unknown architecture {arch!r}. Choose ssdlite or fasterrcnn.")


def load_checkpoint(arch: str, checkpoint: Path | str) -> nn.Module:
    model = build_model(arch, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model
