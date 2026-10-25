"""Regenerate ml/kaggle_finetune_run2.ipynb after editing ml/build_finetune.py or ml/train_detector.py.

    .venv/Scripts/python ml/make_finetune_notebook.py .
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}  # noqa: E731

intro = """# FindMine detector run 2: keys fine-tune (Kaggle)

Starts from run 1's weights and adds keys data, so the detector learns keys without forgetting cups and bottles.
Full steps: `docs/finetune_keys.md`.

1. **Add data** (right panel → *Add Input*):
   - `coco-2017-dataset` (by awsaf49), as in run 1
   - your `findmine-labels` dataset (from run 1)
   - **run 1's notebook output**: *Add Input → Notebooks →* your run-1 notebook (it holds `runs/detect/findmine_run1/weights/best.pt`)
   - each keys export as its own dataset: the Roboflow Universe keys set and our own labelled frames (Roboflow → Export → **YOLOv8** → zip → upload)
2. **Settings:** Accelerator **GPU T4 x1** (or P100), Internet **On**.
3. **Save Version → Save & Run All (Commit)**. About 1.5–3 h; you can close the browser.
4. When it's done: version → **Output** → download `out/findmine12_run2.onnx`, `out/findmine12_run2.json`, `out/run2_report.md`
   into `models/` (the .md into `eval/`), then run `ml/compare_detectors.py` on the laptop.
"""

setup = r'''import glob, os, shutil, subprocess
from pathlib import Path


def find(pattern, depth=6):
    """Matches up to `depth` folders below /kaggle/input. (A recursive ** glob walks all 118k COCO images: minutes.)"""
    return [p for d in range(depth + 1) for p in glob.glob("/kaggle/input/" + "*/" * d + pattern)]


print("Inputs attached:")
for p in sorted(glob.glob("/kaggle/input/*/*/*")):
    print("  ", p)
W = Path("/kaggle/working/findmine")
labels = Path(find("findmine.yaml")[0]).parent
coco = Path(find("train2017/000000000009.jpg")[0]).parent.parent
best1 = find("runs/detect/findmine_run1/weights/best.pt", 4)
extras = sorted({str(Path(p).parent) for p in find("data.yaml") if not p.startswith(str(labels))})
print("labels:", labels, "| coco:", coco, "| run-1 weights:", best1 or "NOT FOUND", "| keys exports:", extras)
assert extras, ("no keys export found (a folder with data.yaml + train/images + train/labels). "
                f"Add the keys dataset under Add Input. Zips seen (Kaggle didn't unpack them?): {find('*.zip')}")

# writable copy of the labels (Ultralytics writes .cache files); never copy image folders
shutil.copytree(labels, W, dirs_exist_ok=True, ignore=shutil.ignore_patterns("images", "*.cache"))
(W / "images").mkdir(exist_ok=True)
for split in ("train2017", "val2017"):
    if not (W / "images" / split).exists():
        os.symlink(coco / split, W / "images" / split)
subprocess.run(["python", "build_finetune.py", "--root", str(W), "--extra", *extras, "--replay", "6000"], check=True)
for lst in ("train_ft.txt", "val_ft.txt"):
    paths = (W / lst).read_text().split()
    missing = [p for p in paths if not (W / p).exists()]
    print(lst, len(paths), "images,", len(missing), "missing")
    assert not missing, missing[:5]
'''

train = r'''# From run 1's weights: 30 epochs, small learning rate, no warm-up. Without them: from stock YOLO11n, 70 epochs.
%cd /kaggle/working
args = ["python", "train_detector.py", "--data", f"{W}/findmine_ft.yaml", "--name", "findmine_run2"]
if best1:
    args += ["--model", best1[0], "--epochs", "30", "--lr0", "0.002", "--warmup-epochs", "0", "--close-mosaic", "5"]
else:
    print("run-1 weights not found: training from stock YOLO11n (slower, still fine overnight)")
    args += ["--model", "yolo11n.pt", "--epochs", "70"]
subprocess.run(args, check=True)
'''

export = r'''import json
from ultralytics import YOLO
run = Path("/kaggle/working/runs/detect/findmine_run2")
best = YOLO(run / "weights/best.pt")
out = Path("/kaggle/working/out"); out.mkdir(exist_ok=True)
onnx = Path(best.export(format="onnx", opset=12, imgsz=640, simplify=True, dynamic=False))   # same as run 1 (cv.dnn)
shutil.copy(onnx, out / "findmine12_run2.onnx")
(out / "findmine12_run2.json").write_text(json.dumps([best.names[i] for i in range(len(best.names))]))
shutil.copy(run / "results.csv", out / "results_run2.csv")

# Per-class AP on the fine-tune val set (run-1 val + held-out keys images), run 1 vs run 2, for the report.
rows = ["| class | AP50 run 1 | AP50 run 2 |", "|---|---|---|"]
m2 = best.val(data=str(W / "findmine_ft.yaml"), imgsz=640, split="val", plots=False).box
m1 = YOLO(best1[0]).val(data=str(W / "findmine_ft.yaml"), imgsz=640, split="val", plots=False).box if best1 else None
for i, name in best.names.items():
    a1 = f"{m1.ap50[list(m1.ap_class_index).index(i)]:.3f}" if m1 is not None and i in m1.ap_class_index else "-"
    a2 = f"{m2.ap50[list(m2.ap_class_index).index(i)]:.3f}" if i in m2.ap_class_index else "-"
    rows.append(f"| {name} | {a1} | {a2} |")
rows.append(f"| **all (mAP50)** | {m1.map50:.3f} |" + f" {m2.map50:.3f} |" if m1 is not None else f"| **all** | - | {m2.map50:.3f} |")
(out / "run2_report.md").write_text("# Detector run 2 (keys fine-tune)\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
'''

nb = {"cells": [md(intro),
                code("!pip install -q ultralytics==8.4.182 onnx==1.23.2 onnxslim==0.1.98"),
                code("%%writefile build_finetune.py\n" + (root / "ml/build_finetune.py").read_text(encoding="utf-8")),
                code("%%writefile train_detector.py\n" + (root / "ml/train_detector.py").read_text(encoding="utf-8")),
                code(setup), code(train), code(export)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(root / "ml/kaggle_finetune_run2.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("written")
