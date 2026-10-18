import json
from pathlib import Path

import cv2
import numpy as np

from vision_detect.data.dataset import CocoPersonSubset
from vision_detect.data.records import (
    license_allows_redistribution,
    person_records_from_coco,
    sanitize_boxes,
)
from vision_detect.paths import SUBSET_PATH

FIXTURE = Path(__file__).parent / "fixtures" / "mini_coco.json"


def test_parser_keeps_only_person_boxes():
    coco = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = person_records_from_coco(coco, split="train")
    assert len(records) == 1
    assert records[0]["id"] == 1
    assert records[0]["boxes"] == [[10.0, 10.0, 30.0, 40.0]]
    assert records[0]["labels"] == [1]
    assert records[0]["license_name"] == "Attribution License"
    assert records[0]["redistributable"] is True


def test_sanitize_drops_collapsed_boxes():
    boxes, labels = sanitize_boxes([[0, 0, 0, 10], [1, 1, 5, 6]], [1, 1], width=10, height=10)
    assert boxes == [[1.0, 1.0, 5.0, 6.0]]
    assert labels == [1]


def test_license_filter():
    assert license_allows_redistribution("Attribution License")
    assert license_allows_redistribution("Attribution-ShareAlike License")
    assert license_allows_redistribution("No known copyright restrictions")
    assert not license_allows_redistribution("Attribution-NonCommercial License")
    assert not license_allows_redistribution("Attribution-NoDerivs License")
    assert not license_allows_redistribution("")


def test_dataset_reads_generated_image(tmp_path: Path):
    image_path = tmp_path / "000000000009.jpg"
    cv2.imwrite(str(image_path), np.full((48, 64, 3), 30, dtype=np.uint8))
    index = {
        "images": [
            {
                "id": 9,
                "file_name": "000000000009.jpg",
                "width": 64,
                "height": 48,
                "split": "train",
                "boxes": [[4, 5, 20, 30]],
                "labels": [1],
            }
        ]
    }
    index_path = tmp_path / "subset.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    dataset = CocoPersonSubset(index_path, tmp_path, "train", training=False)
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 48, 64)
    assert image.dtype == np.dtype("float32") or str(image.dtype) == "torch.float32"
    assert target["boxes"].shape == (1, 4)
    assert int(target["labels"][0]) == 1


def test_committed_subset_is_annotations_only():
    assert SUBSET_PATH.exists(), "run python -m vision_detect prepare"
    document = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    license_name = document["info"]["annotations_license"]
    assert "CC BY" in license_name or "Attribution 4.0" in license_name
    train = [row for row in document["images"] if row["split"] == "train"]
    val = [row for row in document["images"] if row["split"] == "val"]
    assert len(train) >= 50
    assert len(val) >= 20
    assert any(row["redistributable"] for row in val)
    assert SUBSET_PATH.stat().st_size < 2_000_000
    for row in document["images"]:
        assert row["boxes"]
        for box in row["boxes"]:
            assert box[2] > box[0] and box[3] > box[1]
