"""Day-1 spike: export stock YOLO to ONNX, run it with cv.dnn, check it matches PyTorch on 50 COCO val images.

Run from the repo root with the ML venv:  ml/.venv/Scripts/python ml/spike_yolo_dnn.py [yolo11n]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from pipeline.detect import Detector  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"
models, imgdir = ROOT / "models", ROOT / "data/coco/val50"
models.mkdir(exist_ok=True)
imgdir.mkdir(parents=True, exist_ok=True)

# 1. Export (opset 12, static 640x640, simplified), as the plan says
pt = YOLO(str(models / f"{name}.pt"))
onnx_path = Path(pt.export(format="onnx", opset=12, imgsz=640, simplify=True, dynamic=False))
onnx_path.with_suffix(".json").write_text(json.dumps([pt.names[i] for i in range(len(pt.names))]))
print("cv2", cv.__version__, "| onnx", onnx_path)

# 2. First 50 COCO val2017 images (by id)
meta = json.loads((ROOT / "data/coco/annotations/instances_val2017.json").read_text())
images = sorted(meta["images"], key=lambda im: im["id"])[:50]
for im in images:
    f = imgdir / im["file_name"]
    if not f.exists():
        urllib.request.urlretrieve(im["coco_url"], f)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# 3. Compare: every PyTorch box should have a cv.dnn box of the same class with IoU > 0.9
det = Detector(str(onnx_path))
matched = total = extra = 0
times = []
for im in images:
    img = cv.imread(str(imgdir / im["file_name"]))
    t = time.perf_counter()
    ours = det(img)
    times.append(time.perf_counter() - t)
    r = pt.predict(img, imgsz=640, rect=False, conf=0.25, iou=0.45, verbose=False)[0].boxes
    ref = [(int(c), *xyxy) for c, xyxy in zip(r.cls.tolist(), r.xyxy.tolist())]
    total += len(ref)
    used = set()
    for c, *box in ref:
        best = max(((iou(box, d[2:]), j) for j, d in enumerate(ours) if d[0] == c and j not in used), default=(0, -1))
        if best[0] > 0.9:
            matched += 1
            used.add(best[1])
    extra += len(ours) - len(used)

ms = 1000 * float(np.median(times[1:]))
print(f"PyTorch boxes: {total}  matched by cv.dnn: {matched} ({matched / total:.1%})  extra cv.dnn boxes: {extra}")
# PyTorch shares the CPU threads in this process, so this is slower than the server (~65 ms on the dev laptop)
print(f"cv.dnn median {ms:.1f} ms/frame (CPU, incl. pre/post-processing)")
assert matched / total >= 0.95, "cv.dnn output does not match PyTorch"
print("SPIKE PASSED")
