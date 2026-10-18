"""A small crop classifier used as a CNN baseline next to the detector.

It answers "is this window a person?" and does not emit boxes. The training
script reports accuracy so the README can contrast that number with detection mAP.
"""

from __future__ import annotations

from torch import Tensor, nn


class CropCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images).flatten(1))
