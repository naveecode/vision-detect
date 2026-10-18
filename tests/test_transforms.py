import albumentations as A
import numpy as np

from vision_detect.data.transforms import augment_detection, horizontal_flip, opencv_brightness


def test_brightness_identity_and_shift():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert np.array_equal(opencv_brightness(image, 0), image)
    assert int(opencv_brightness(image, 20)[0, 0, 0]) == 120


def test_horizontal_flip_mirrors_boxes():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :10] = 255
    flipped, boxes = horizontal_flip(image, [[10, 5, 30, 25]])
    assert boxes == [[50, 5, 70, 25]]
    assert flipped[0, -1, 0] == 255


def test_eval_augmentation_is_identity():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)
    boxes = [[1.0, 2.0, 8.0, 9.0]]
    labels = [1]
    out_image, out_boxes, out_labels = augment_detection(image, boxes, labels, training=False)
    assert out_boxes == boxes
    assert out_labels == labels
    assert np.array_equal(out_image, image)


def test_albumentations_flip_updates_boxes():
    image = np.zeros((50, 80, 3), dtype=np.uint8)
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    result = transform(image=image, bboxes=[[10, 5, 30, 40]], labels=[1])
    x1, _y1, x2, _y2 = result["bboxes"][0]
    assert abs(x1 - 50) < 1e-3
    assert abs(x2 - 70) < 1e-3
    assert result["labels"] == [1]
