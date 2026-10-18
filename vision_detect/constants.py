"""Shared detection constants. No heavy imports, so the API can load this module."""

import math

INPUT_SIZE = 320
PERSON_INDEX = 1
PERSON_NAME = "person"
NMS_IOU = 0.55
TOPK = 300
DEFAULT_SCORE_THRESHOLD = 0.4
# torchvision SSD box coder weights and the exp() clamp used by BoxCoder.
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
BBOX_XFORM_CLIP = math.log(1000.0 / 16)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
# Official annotations zip. The custom domain's TLS certificate is issued for
# s3.amazonaws.com, so the path-style URL is the one that verifies.
ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATIONS_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
IMAGE_URL = "https://s3.amazonaws.com/images.cocodataset.org/{split}2017/{file_name}"
