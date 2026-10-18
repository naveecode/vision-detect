"""Export the person SSDLite head to ONNX and check it against PyTorch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from vision_detect.constants import (
    DEFAULT_SCORE_THRESHOLD,
    INPUT_SIZE,
    NMS_IOU,
    PERSON_NAME,
    TOPK,
)
from vision_detect.models.exportable import build_exportable
from vision_detect.paths import DEFAULT_METADATA_PATH, DEFAULT_MODEL_PATH
from vision_detect.utils.boxes import box_iou, postprocess

MAX_ONNX_BYTES = 25 * 1024 * 1024


def export_onnx(module, example: torch.Tensor, path: Path, output_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(
        module,
        example,
        str(path),
        input_names=["images"],
        output_names=output_names,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )


def max_abs_error(module, path: Path, example: torch.Tensor) -> float:
    module.eval()
    with torch.no_grad():
        pytorch_out = module(example)
    if isinstance(pytorch_out, torch.Tensor):
        pytorch_arrays = [pytorch_out.detach().cpu().numpy()]
    else:
        pytorch_arrays = [tensor.detach().cpu().numpy() for tensor in pytorch_out]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {session.get_inputs()[0].name: example.detach().cpu().numpy()}
    onnx_arrays = session.run(None, feeds)
    gaps = [
        float(np.max(np.abs(left - right)))
        for left, right in zip(pytorch_arrays, onnx_arrays, strict=True)
    ]
    return max(gaps)


def _maybe_quantize(path: Path) -> Path:
    if path.stat().st_size <= MAX_ONNX_BYTES:
        return path
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantized = path.with_name(path.stem + ".int8.onnx")
    quantize_dynamic(str(path), str(quantized), weight_type=QuantType.QUInt8)
    if quantized.stat().st_size < path.stat().st_size:
        quantized.replace(path)
    else:
        quantized.unlink(missing_ok=True)
    return path


@torch.no_grad()
def compare_detections(detector, exportable, image: torch.Tensor, threshold: float) -> dict:
    """Compare torchvision postprocess with the exported head on one CHW image in ``[0, 1]``."""
    import torch.nn.functional as F

    detector.eval()
    detector.score_thresh = threshold
    detector.nms_thresh = NMS_IOU
    official = detector([image])[0]
    resized = F.interpolate(
        image[None], size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
    )
    boxes, scores = exportable(resized)
    height, width = int(image.shape[1]), int(image.shape[2])
    kept_boxes, kept_scores = postprocess(
        boxes[0].cpu().numpy(),
        scores[0].cpu().numpy(),
        width,
        height,
        threshold,
        NMS_IOU,
        INPUT_SIZE,
    )
    person = official["labels"] == 1
    official_boxes = official["boxes"][person]
    official_scores = official["scores"][person]
    result = {
        "official": int(len(official_scores)),
        "exported": int(len(kept_scores)),
        "top_iou": None,
        "top_score_abs": None,
    }
    if len(official_scores) == 0 or len(kept_scores) == 0:
        return result
    result["top_iou"] = float(box_iou(official_boxes[0].tolist(), kept_boxes[0].tolist()))
    result["top_score_abs"] = abs(float(official_scores[0]) - float(kept_scores[0]))
    return result


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def export_detector(
    detector,
    output: Path,
    metadata_path: Path,
    model_version: str,
    dataset: str,
    reference_image: Path | None = None,
) -> dict:
    exportable = build_exportable(detector)
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    export_onnx(exportable, example, output, ["boxes", "scores"])
    _maybe_quantize(output)
    numeric = max_abs_error(exportable, output, example)
    if numeric > 1e-3:
        raise RuntimeError(f"ONNX numeric drift {numeric} exceeds 1e-3")
    agreement: dict = {"max_abs_error": numeric}
    if reference_image is not None:
        photo = compare_detections(
            detector, exportable, _load_rgb_tensor(reference_image), threshold=0.3
        )
        photo["image"] = reference_image.name
        if photo["top_iou"] is None or photo["top_iou"] < 0.99 or photo["top_score_abs"] > 1e-3:
            raise RuntimeError(f"PyTorch postprocess mismatch: {photo}")
        agreement["validation_photo"] = photo
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "model_name": "ssdlite320_mobilenet_v3_large",
        "model_version": model_version,
        "task": "person detection",
        "dataset": dataset,
        "input_name": "images",
        "input_size": INPUT_SIZE,
        "input_layout": "RGB NCHW float32 in [0, 1]",
        "input_note": (
            "Image is stretched to 320x320. Normalization to [-1, 1] is inside the graph."
        ),
        "outputs": ["boxes", "scores"],
        "boxes_format": "xyxy in the 320x320 input, top-k before NMS, sorted by score",
        "class_name": PERSON_NAME,
        "person_index": 1,
        "topk": TOPK,
        "nms_iou": NMS_IOU,
        "default_score_threshold": DEFAULT_SCORE_THRESHOLD,
        "opset": 17,
        "sha256": digest,
        "bytes": output.stat().st_size,
        "parity": agreement,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    summary = {"onnx": str(output), "bytes": metadata["bytes"], "parity": metadata["parity"]}
    print(json.dumps(summary))
    return metadata


def export_command(args) -> dict:
    from vision_detect.models.detector import build_model, load_checkpoint

    if args.checkpoint:
        detector = load_checkpoint(args.arch, args.checkpoint)
        version = args.version or f"{args.arch}-{Path(args.checkpoint).stem}"
    else:
        detector = build_model(args.arch, pretrained=True)
        version = args.version or f"{args.arch}-coco-pretrained"
    detector.eval()
    return export_detector(
        detector,
        Path(args.output or DEFAULT_MODEL_PATH),
        Path(args.metadata or DEFAULT_METADATA_PATH),
        version,
        args.dataset,
        reference_image=getattr(args, "reference_image", None),
    )
