"""Face blur (F8.2): faces found in a real photo are gone after blurring; a photo with no face is unchanged.

Run from the repo root:  .venv/Scripts/python server/test_privacy.py
"""
from pathlib import Path

import cv2 as cv

from pipeline.privacy import FaceBlur

ROOT = Path(__file__).resolve().parent.parent
blur = FaceBlur(ROOT / "models/face_detection_yunet_2023mar.onnx")

for name in ("000000005001.jpg", "000000001000.jpg", "000000002685.jpg"):  # COCO photos with several faces
    img = cv.imread(str(ROOT / "data/coco/val50" / name))
    before = len(blur.faces(img))
    out, n = blur(img)
    after = len(blur.faces(out))
    print(f"{name}: {before} faces -> {after} after blurring")
    assert n >= before > 0 and after == 0, (name, before, after)
    assert out.shape == img.shape

ref = next((ROOT / "data/home/cup__my_cup/ref").glob("*.jpg"))
img = cv.imread(str(ref))
out, n = blur(img)
assert n == 0 and (out == img).all(), n   # item photo without a face: stored as is
print("PRIVACY TEST PASSED")
