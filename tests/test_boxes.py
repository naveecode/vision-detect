import numpy as np

from vision_detect.utils.boxes import box_iou, clip_boxes, nms, postprocess, scale_boxes


def test_iou_identical_and_disjoint():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    # Intersection 5*10=50, union 100+100-50=150.
    assert abs(box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-6


def test_nms_keeps_highest_and_disjoint():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_nms_empty():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5).size == 0


def test_scale_full_frame():
    boxes = np.array([[0, 0, 320, 320]], dtype=np.float32)
    scaled = scale_boxes(boxes, orig_width=640, orig_height=480, input_size=320)
    assert np.allclose(scaled, [[0, 0, 640, 480]])


def test_clip_boxes():
    boxes = np.array([[-5, -1, 50, 10]], dtype=np.float32)
    clipped = clip_boxes(boxes, width=40, height=8)
    assert np.allclose(clipped, [[0, 0, 40, 8]])


def test_postprocess_threshold_and_scale():
    boxes = np.array([[0, 0, 160, 320], [10, 10, 20, 20]], dtype=np.float32)
    scores = np.array([0.9, 0.1], dtype=np.float32)
    kept_boxes, kept_scores = postprocess(
        boxes, scores, orig_width=640, orig_height=320, score_threshold=0.4, iou_threshold=0.5
    )
    assert len(kept_scores) == 1
    assert abs(kept_scores[0] - 0.9) < 1e-6
    assert np.allclose(kept_boxes[0], [0, 0, 320, 320])
