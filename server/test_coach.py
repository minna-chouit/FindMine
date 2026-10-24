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


def hint(frame, coach=None, now=0.0):
    return (coach or Coach())(frame, now)["hint"]


assert issue(good) is None and hint(good) is None
blurry = cv.GaussianBlur(good, (0, 0), 6)
assert issue(blurry) is None and hint(blurry) == "blur"     # still detected on: say the fix, keep the frame
assert issue((good * 0.12).astype(np.uint8)) == "dark"
assert issue(cv.convertScaleAbs(good, alpha=3, beta=120)) == "bright"

moved = cv.warpAffine(good, np.float32([[1, 0, 60], [0, 1, 0]]), (w, h))
c = Coach()
c(good, 0.0)
r = c(moved, 0.2)
assert r["issue"] is None and r["hint"] == "shake", r      # fast sweep, sharp frame: hint only
c = Coach()
c(good, 0.0)
assert issue(cv.GaussianBlur(moved, (0, 0), 6), c, 0.2) == "shake"   # moving and smeared: skip it
c = Coach()
c(good, 0.0)
r = c(cv.warpAffine(good, np.float32([[1, 0, 10], [0, 1, 0]]), (w, h)), 0.2)
assert r["issue"] is None and r["hint"] is None, r          # slow sweep is fine

# Finger on the lens: dark, flat, reddish. Dark at first, "blocked" once it lasts >= 1 s, then "fixed".
finger = np.full_like(good, (20, 30, 70))
c = Coach()
assert issue(finger, c, 0.0) == "dark"
assert issue(finger, c, 1.2) == "blocked"
r = c(good, 1.4)
assert r["issue"] is None and r["fixed"] == "blocked", r
print("COACH TEST PASSED")
