"""Traceable SSDLite wrapper.

The ONNX graph returns the top person-class boxes before NMS. NMS stays in
Python and in the browser so both sides share one implementation, and so the
graph stays free of the control flow in torchvision's postprocess.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vision_detect.constants import BBOX_XFORM_CLIP, BOX_WEIGHTS, INPUT_SIZE, PERSON_INDEX, TOPK


def decode_boxes(rel_codes: Tensor, anchors: Tensor) -> Tensor:
    """Decode SSD box deltas. Matches ``BoxCoder.decode_single`` for weights (10, 10, 5, 5)."""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    wx, wy, ww, wh = BOX_WEIGHTS
    dx = rel_codes[..., 0] / wx
    dy = rel_codes[..., 1] / wy
    dw = torch.clamp(rel_codes[..., 2] / ww, max=BBOX_XFORM_CLIP)
    dh = torch.clamp(rel_codes[..., 3] / wh, max=BBOX_XFORM_CLIP)
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    return torch.stack(
        (
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ),
        dim=-1,
    )


class PersonSSDLite(nn.Module):
    """SSDLite person head. Input is RGB float in ``[0, 1]``, shape ``[N, 3, 320, 320]``."""

    def __init__(self, detector: nn.Module, anchors: Tensor, topk: int = TOPK) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.head = detector.head
        self.register_buffer("anchors", anchors)
        self.topk = topk

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (images - 0.5) / 0.5
        features = self.backbone(normalized)
        feature_list = [features[str(index)] for index in range(len(features))]
        outputs = self.head(feature_list)
        boxes = decode_boxes(outputs["bbox_regression"], self.anchors).clamp(0.0, float(INPUT_SIZE))
        scores = torch.softmax(outputs["cls_logits"], dim=-1)[..., PERSON_INDEX]
        k = min(self.topk, int(scores.shape[1]))
        top_scores, index = torch.topk(scores, k=k, dim=1)
        top_boxes = torch.gather(boxes, 1, index.unsqueeze(-1).expand(-1, -1, 4))
        return top_boxes, top_scores


@torch.no_grad()
def compute_anchors(detector: nn.Module) -> Tensor:
    detector.eval()
    image = torch.zeros(3, INPUT_SIZE, INPUT_SIZE)
    transformed, _ = detector.transform([image], None)
    features = detector.backbone(transformed.tensors)
    feature_list = list(features.values())
    anchors = detector.anchor_generator(transformed, feature_list)[0]
    return anchors.detach().cpu()


def build_exportable(detector: nn.Module, topk: int = TOPK) -> PersonSSDLite:
    module = PersonSSDLite(detector, compute_anchors(detector), topk=topk)
    module.eval()
    return module
