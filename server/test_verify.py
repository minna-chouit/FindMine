"""Verifier ranks the same object (new angle, darker) above a look-alike of the same class.

Run from the repo root:  .venv/Scripts/python server/test_verify.py
"""
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.verify import Verifier

ROOT = Path(__file__).resolve().parent.parent
img = cv.imread(str(ROOT / "data/coco/val50/000000000139.jpg"))
tv, other_tv = img[167:262, 6:155], img[208:333, 554:640]  # two TVs in the same room (YOLO boxes)

v = Verifier(ROOT / "models/dinov2_vits14.onnx")
refs = [v.features(tv), v.features(cv.flip(tv, 1))]
h, w = tv.shape[:2]
again = cv.warpPerspective(tv, cv.getPerspectiveTransform(
    np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32([[6, 4], [w - 2, 0], [w, h - 5], [3, h]])), (w, h))
same, lookalike = v(v.features((again * 0.7).astype(np.uint8)), refs), v(v.features(other_tv), refs)
print(f"same item {same:.3f}  look-alike {lookalike:.3f}")
assert 0 <= lookalike < same <= 1
assert lookalike < 0.8, "look-alike would be announced as found"
print("VERIFY TEST PASSED")
