"""Tile search finds a small bottle the plain pass misses; the tracker follows a moving object;
Kalman smoothing beats noisy boxes; distance bands make sense.

Run from the repo root:  .venv/Scripts/python server/test_track.py
"""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

from pipeline.detect import Detector
from pipeline.guide import distance
from pipeline.track import Track

ROOT = Path(__file__).resolve().parent.parent
names = json.loads((ROOT / "models/findmine12.json").read_text())
det = Detector(str(ROOT / "models/findmine12.onnx"))

# Tile search: shrink a bottle scene to half size in the corner of another room.
scene = cv.imread(str(ROOT / "data/coco/val50/000000000632.jpg"))
canvas = cv.resize(cv.imread(str(ROOT / "data/coco/val50/000000000139.jpg")), scene.shape[1::-1])
small = cv.resize(scene, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
canvas[:small.shape[0], :small.shape[1]] = small
bottles = lambda ds: [d for d in ds if names[d[0]] == "bottle"]  # noqa: E731
assert not bottles(det(canvas)), "plain pass should miss the small bottle (test premise)"
found = bottles(det.tiles(canvas))
assert found and found[0][3] < small.shape[0], found  # found, and inside the shrunk corner
print("tiles: plain 0 bottles, tiled", [round(d[1], 2) for d in found])

# Tracker: a textured patch moves 8 px right per frame; detector "misses" every frame.
rng = np.random.default_rng(0)
bg = rng.integers(0, 60, (240, 320, 3), np.uint8)
patch = cv.resize(scene[150:230, 300:380], (60, 60))


def frame(x):
    f = bg.copy()
    f[90:150, x:x + 60] = patch
    return f


t = Track(frame(40), (40, 90, 100, 150))
for i in range(1, 9):
    box = t.update(frame(40 + 8 * i))
cx = (box[0] + box[2]) / 2
assert abs(cx - (40 + 8 * 8 + 30)) < 12, box   # follows the patch (true centre 134)

# Kalman: noisy detections (+-8 px) of a still box -> smoothed centre wobbles much less.
t = Track(frame(100), (100, 90, 160, 150))
raw, smooth = [], []
for _ in range(30):
    n = rng.normal(0, 8, 4)
    d = (100 + n[0], 90 + n[1], 160 + n[2], 150 + n[3])
    raw.append((d[0] + d[2]) / 2)
    b = t.update(frame(100), d)
    smooth.append((b[0] + b[2]) / 2)
assert np.std(smooth[5:]) < 0.6 * np.std(raw[5:]), (np.std(raw), np.std(smooth))
print(f"kalman: raw centre std {np.std(raw[5:]):.1f} px -> smoothed {np.std(smooth[5:]):.1f} px")

# Tracker gives up after repeated failures: object vanishes into a flat image.
t = Track(frame(40), (40, 90, 100, 150))
flat = np.zeros_like(bg)
assert any(t.update(flat) is None for _ in range(30)), "track should be dropped"

# Distance: a 10 cm cup 100 px wide in a 640 px frame (65 deg FOV) is ~0.5 m away.
assert distance("cup", (270, 200, 370, 280), 640) == ("arm", 0.5)
assert distance("cup", (120, 100, 520, 400), 640)[0] == "near"
assert distance("bag", (300, 220, 340, 260), 640)[0] == "far"
print("TRACK TEST PASSED")
