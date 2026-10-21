"""Camera coach on synthetic bad frames made from a real COCO photo.

Run from the repo root:  .venv/Scripts/python server/test_coach.py
"""
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.coach import Coach

ROOT = Path(__file__).resolve().parent.parent
good = cv.imread(str(ROOT / "data/coco/val50/000000000139.jpg"))
h, w = good.shape[:2]


def issue(frame, coach=None, now=0.0):
    return (coach or Coach())(frame, now)["issue"]


assert issue(good) is None
assert issue(cv.GaussianBlur(good, (0, 0), 6)) == "blur"
assert issue((good * 0.12).astype(np.uint8)) == "dark"
assert issue(cv.convertScaleAbs(good, alpha=3, beta=120)) == "bright"

c = Coach()
c(good, 0.0)
assert issue(cv.warpAffine(good, np.float32([[1, 0, 60], [0, 1, 0]]), (w, h)), c, 0.2) == "shake"
c = Coach()
c(good, 0.0)
assert issue(cv.warpAffine(good, np.float32([[1, 0, 10], [0, 1, 0]]), (w, h)), c, 0.2) is None  # slow sweep is fine

# Finger on the lens: dark, flat, reddish. Dark at first, "blocked" once it lasts >= 1 s, then "fixed".
finger = np.full_like(good, (20, 30, 70))
c = Coach()
assert issue(finger, c, 0.0) == "dark"
assert issue(finger, c, 1.2) == "blocked"
r = c(good, 1.4)
assert r["issue"] is None and r["fixed"] == "blocked", r
print("COACH TEST PASSED")
