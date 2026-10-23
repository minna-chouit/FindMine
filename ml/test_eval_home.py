"""eval_home on a tiny fake home set made of two different COCO cups, each a look-alike of the other:
alone photos + a short video, one "with_other" photo (both cups in view), then a --fit calibration.

Run from the repo root:  .venv/Scripts/python ml/test_eval_home.py
"""
import json
import tempfile
from pathlib import Path

import cv2 as cv

import eval_home
from eval_home import ROOT, evaluate, fit, pick_thresholds, summary
from pipeline.detect import Detector
from pipeline.verify import Verifier

assert pick_thresholds([0.9, 0.8, 0.6, 0.3], [1, 1, 0, 0]) == (0.61, 0.61)
assert pick_thresholds([0.9, 0.7, 0.6, 0.3], [1, 1, 1, 0]) == (0.5, 0.3)    # floors, not 0.05

names = json.loads((ROOT / "models/findmine12.json").read_text())
det, ver = Detector(str(ROOT / "models/findmine12.onnx")), Verifier(ROOT / "models/dinov2_vits14.onnx")
mine = cv.imread(str(ROOT / "data/coco/val50/000000002592.jpg"))
other = cv.imread(str(ROOT / "data/coco/val50/000000003845.jpg"))

with tempfile.TemporaryDirectory() as home:
    for item, img in (("cup__mine", mine), ("cup__other", other)):
        _, _, x1, y1, x2, y2 = max((d for d in det(img) if names[d[0]] == "cup"), key=lambda d: d[1])
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        (Path(home) / item / "ref").mkdir(parents=True)
        (Path(home) / item / "test" / "with_other").mkdir(parents=True)
        cv.imwrite(f"{home}/{item}/ref/1.jpg", crop)
        cv.imwrite(f"{home}/{item}/ref/2.jpg", cv.flip(crop, 1))
        cv.imwrite(f"{home}/{item}/test/room.jpg", img)
    vid = cv.VideoWriter(f"{home}/cup__mine/test/clip.avi", cv.VideoWriter_fourcc(*"MJPG"), 10, mine.shape[1::-1])
    for _ in range(10):  # 1 s video of my cup's room
        vid.write(mine)
    vid.release()
    side = cv.hconcat([cv.resize(other, (mine.shape[1], mine.shape[0])), mine])  # both cups in view, any layout
    cv.imwrite(f"{home}/cup__mine/test/with_other/both.jpg", side)

    rows = evaluate(home, det, names, ver, fps=10)

ref_rows = [r for r in rows if r["setting"] == "ref"]   # each ref photo vs the item's other refs / the look-alike's
assert len(ref_rows) == 8 and all(r["match"] > s["match"] for r in ref_rows if r["same_item"]
                                  for s in ref_rows if not s["same_item"] and s["item"] == r["item"]), ref_rows
rows_all, rows = rows, [r for r in rows if r["setting"] != "ref"]
own = [r for r in rows if r["same_item"]]
cross = [r for r in rows if not r["same_item"]]
assert {(r["item"], r["setting"]) for r in own} == {("cup__mine", "alone"), ("cup__mine", "with_other"),
                                                     ("cup__other", "alone")}
assert len(own) == 13 and len(cross) == 12, (len(own), len(cross))   # with_other frame: positive only
both = [r for r in rows if r["setting"] == "with_other"]
assert len(both) == 1 and both[0]["same_item"] and both[0]["match"] >= 0.8, both   # my cup found among the two
assert all(r["detected"] and r["emb"] > 0 for r in own)

print(summary(rows, 0.8, 0.5).encode("ascii", "replace").decode())
assert all(r["match"] < 0.8 for r in cross) and "0 of 12 |" in summary(rows, 0.8, 0.5)

with tempfile.TemporaryDirectory() as tmp:
    eval_home.FUSION = Path(tmp) / "fusion.json"                     # never touch the real calibration here
    thetas, found, ask = fit(rows_all)
    saved = json.loads(eval_home.FUSION.read_text())
assert saved["found"] == found and saved["ask"] == ask and len(saved["thetas"]) == 4 and 0.5 <= found <= 0.95
ver.w = thetas
refit = [(r["same_item"], ver.fuse([r["emb"], r["geom"], r["colour"]])) for r in rows_all]
assert all(s < found for same, s in refit if not same), refit           # no look-alike reaches "found"
print(f"fit: found {found}, ask {ask}")
print("EVAL HOME TEST PASSED")
