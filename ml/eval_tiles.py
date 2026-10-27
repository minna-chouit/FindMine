"""Detection of one class (default keys) on FindMine-Home, the way the server searches for small items: close-up reference photos
(conf 0.25), search frames with 2x2 zoomed tiles (conf 0.05 and 0.25), and false boxes of that class on every
other item's frames (tiles, conf 0.25). Numbers in eval/run2_report.md and eval/run3_report.md.

    .venv/Scripts/python ml/eval_tiles.py models/findmine12.onnx models/findmine12_run3.onnx [--cls wallet]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "server"), str(ROOT / "ml")]
from eval_home import IMG, frames  # noqa: E402
from pipeline.detect import Detector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--cls", default="keys")
a = ap.parse_args()
home = ROOT / "data/home"
items = [d for d in sorted(home.iterdir()) if d.is_dir() and "__" in d.name]
test = lambda ds: [img for d in ds for f in (d / "test", d / "test/with_other")  # noqa: E731
                   for _, img in frames(f, 2) if img is not None]
keys = [d for d in items if d.name.startswith(a.cls + "__")]
refs = [cv.imread(str(p)) for d in keys for p in sorted((d / "ref").glob("*")) if p.suffix.lower() in IMG]
kf, of = test(keys), test([d for d in items if d not in keys])
for model in a.models:
    KEYS = json.loads(Path(model).with_suffix(".json").read_text()).index(a.cls)
    d25, d05 = Detector(model), Detector(model, conf=0.05)
    has = lambda dets: any(c == KEYS for c, *_ in dets)  # noqa: E731
    print(f"{Path(model).name}: refs {sum(has(d25(i)) for i in refs)}/{len(refs)} | {a.cls} frames, tiles "
          f"@0.05 {sum(has(d05.tiles(i)) for i in kf)}/{len(kf)}, @0.25 {sum(has(d25.tiles(i)) for i in kf)}/{len(kf)}"
          f" | false {a.cls} on other frames, tiles @0.25 {sum(has(d25.tiles(i)) for i in of)}/{len(of)}")
