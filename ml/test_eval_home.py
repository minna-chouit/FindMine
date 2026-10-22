"""eval_home on a tiny fake home set: two different cups (one photo + one short video), each a look-alike of the other.

Run from the repo root:  .venv/Scripts/python ml/test_eval_home.py
"""
import json
import tempfile
from pathlib import Path

import cv2 as cv

from eval_home import ROOT, evaluate, summary
from pipeline.detect import Detector
from pipeline.verify import Verifier

names = json.loads((ROOT / "models/findmine12.json").read_text())
det, ver = Detector(str(ROOT / "models/findmine12.onnx")), Verifier(ROOT / "models/dinov2_vits14.onnx")

with tempfile.TemporaryDirectory() as home:
    for item, photo in (("cup__mine", "000000002592"), ("cup__other", "000000003845")):
        img = cv.imread(str(ROOT / f"data/coco/val50/{photo}.jpg"))
        _, _, x1, y1, x2, y2 = max((d for d in det(img) if names[d[0]] == "cup"), key=lambda d: d[1])
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        (Path(home) / item / "ref").mkdir(parents=True)
        (Path(home) / item / "test").mkdir()
        cv.imwrite(f"{home}/{item}/ref/1.jpg", crop)
        cv.imwrite(f"{home}/{item}/ref/2.jpg", cv.flip(crop, 1))
        cv.imwrite(f"{home}/{item}/test/room.jpg", img)
    mine = cv.imread(str(ROOT / "data/coco/val50/000000002592.jpg"))
    vid = cv.VideoWriter(f"{home}/cup__mine/test/clip.avi", cv.VideoWriter_fourcc(*"MJPG"), 10, mine.shape[1::-1])
    for _ in range(10):  # 1 s video of my cup's room
        vid.write(mine)
    vid.release()

    rows = evaluate(home, det, names, ver, fps=2)

frames = {(r["item"], r["frame"]) for r in rows}
assert frames == {("cup__mine", "room.jpg"), ("cup__mine", "clip.avi#0"), ("cup__mine", "clip.avi#5"),
                  ("cup__other", "room.jpg")}, frames                      # 10 frames at 10 fps, sampled at 2 fps
own = [r for r in rows if r["same_item"]]
cross = [r for r in rows if not r["same_item"]]
assert len(own) == 4 and len(cross) == 4
assert all(r["detected"] and r["decision"] == "found" for r in own), own
assert all(r["decision"] != "found" for r in cross), cross               # never "found" for the other cup
md = summary(rows)
print(md)
assert "| cup__mine | 3 |" in md and "0 of 4 |" in md
print("EVAL HOME TEST PASSED")
