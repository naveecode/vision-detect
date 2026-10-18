import torch

from vision_detect.models.cnn import CropCNN
from vision_detect.models.detector import build_model


def test_crop_cnn_shape():
    model = CropCNN().eval()
    logits = model(torch.rand(2, 3, 64, 64))
    assert tuple(logits.shape) == (2, 2)


def test_ssdlite_forward_without_pretrained_weights():
    model = build_model("ssdlite", pretrained=False)
    model.eval()
    output = model([torch.rand(3, 320, 320)])
    assert {"boxes", "scores", "labels"} <= set(output[0])
    assert output[0]["boxes"].ndim == 2


def test_unknown_arch():
    try:
        build_model("yolo")
    except ValueError as exc:
        assert "ssdlite" in str(exc)
    else:
        raise AssertionError("expected ValueError")
