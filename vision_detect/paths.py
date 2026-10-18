"""Filesystem locations for the subset index, checkpoints, and the shipped ONNX model."""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SUBSET_PATH = PACKAGE_ROOT / "data" / "coco_person_subset.json"
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "coco-person" / "images"
DEFAULT_CHECKPOINT = REPO_ROOT / "outputs" / "ssdlite_person.pth"
DEFAULT_MODEL_PATH = REPO_ROOT / "web" / "model" / "detector.onnx"
DEFAULT_METADATA_PATH = REPO_ROOT / "web" / "model" / "metadata.json"


def model_path() -> Path:
    override = os.environ.get("VISION_DETECT_MODEL")
    return Path(override) if override else DEFAULT_MODEL_PATH


def metadata_path() -> Path:
    override = os.environ.get("VISION_DETECT_METADATA")
    return Path(override) if override else DEFAULT_METADATA_PATH
