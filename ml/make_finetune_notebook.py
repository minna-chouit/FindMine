"""Regenerate ml/kaggle_finetune_run3.ipynb after editing ml/build_finetune.py or ml/train_detector.py.

    .venv/Scripts/python ml/make_finetune_notebook.py .

Run N fine-tunes from run N-1's best.pt (attached as a Kaggle notebook input). ml/kaggle_finetune_run2.ipynb is
the record of run 2 (keys) and is no longer regenerated.
"""
import json
import sys
from pathlib import Path

RUN, PREV = 3, 2
root = Path(sys.argv[1])
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}  # noqa: E731
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}  # noqa: E731
fill = lambda s: s.replace("RUNN", str(RUN)).replace("PREVV", str(PREV))  # noqa: E731

intro = fill("""# FindMine detector run RUNN: wallet, earbuds, medicine fine-tune (Kaggle)

Starts from run PREVV's weights (which learned keys) and adds Roboflow Universe sets for the weak classes, with the
keys set kept in so keys don't regress. Full steps: `docs/finetune_run3.md`.

1. **Add data** (right panel → *Add Input*):
   - `coco-2017-dataset` (by awsaf49), as in runs 1 and 2
   - your `findmine-labels` dataset (from run 1)
   - **run PREVV's notebook output**: *Add Input → Notebooks →* your run-PREVV notebook (it holds `runs/detect/findmine_runPREVV/weights/best.pt`)
   - the keys dataset from run 2 (`keys-computer-vision`) and each new export as its own dataset (Roboflow → Export → **YOLOv8** → zip → upload)
2. **Settings:** Accelerator **GPU T4 x1** (or P100), Internet **On**.
3. **Save Version → Save & Run All (Commit)**. About 2–3 h; you can close the browser.
4. When it's done: version → **Output** → download `out/findmine12_runRUNN.onnx`, `out/findmine12_runRUNN.json`, `out/runRUNN_report.md`
   into `models/` (the .md into `eval/`), then run `ml/compare_detectors.py` on the laptop.
""")

setup = fill(r'''import glob, os, shutil, subprocess
from pathlib import Path


def find(pattern, depth=6):
    """Matches up to `depth` folders below /kaggle/input. (A recursive ** glob walks all 118k COCO images: minutes.)"""
    return [p for d in range(depth + 1) for p in glob.glob("/kaggle/input/" + "*/" * d + pattern)]


def run(args):
    """Run a script and print its output live: a child process's output never reaches Kaggle's log otherwise."""
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
    if p.returncode:
        raise SystemExit(f"{args[1]} failed (exit {p.returncode}): see the lines above")



print("Inputs attached:")
for p in sorted(glob.glob("/kaggle/input/*/*/*")):
    print("  ", p)
W = Path("/kaggle/working/findmine")
# the run-1 labels dataset; run PREVV's output also has a findmine/ copy (it has findmine_ft.yaml): skip that one
labels = Path(next(p for p in find("findmine.yaml") if not os.path.exists(Path(p).parent / "findmine_ft.yaml"))).parent
coco = Path(find("train2017/000000000009.jpg")[0]).parent.parent
base = find("runs/detect/findmine_runPREVV/weights/best.pt", 4)
extras = sorted({str(Path(p).parent) for p in find("data.yaml") if not p.startswith(str(labels))})
print("labels:", labels, "| coco:", coco, "| run-PREVV weights:", base or "NOT FOUND", "| exports:", extras)
assert base, "run PREVV's best.pt not found: Add Input → Notebooks → your run-PREVV notebook"
assert extras, ("no export found (a folder with data.yaml + train/images + train/labels). "
                f"Add the datasets under Add Input. Zips seen (Kaggle didn't unpack them?): {find('*.zip')}")

# writable copy of the labels (Ultralytics writes .cache files); never copy image folders
shutil.copytree(labels, W, dirs_exist_ok=True, ignore=shutil.ignore_patterns("images", "*.cache"))
(W / "images").mkdir(exist_ok=True)
for split in ("train2017", "val2017"):
    if not (W / "images" / split).exists():
        os.symlink(coco / split, W / "images" / split)
run(["python", "build_finetune.py", "--root", str(W), "--extra", *extras, "--replay", "6000"])
for lst in ("train_ft.txt", "val_ft.txt"):
    paths = (W / lst).read_text().split()
    missing = [p for p in paths if not (W / p).exists()]
    print(lst, len(paths), "images,", len(missing), "missing")
    assert not missing, missing[:5]
''')

train = fill(r'''# From run PREVV's weights: 30 epochs, small learning rate, no warm-up (same recipe as run 2).
%cd /kaggle/working
run(["python", "train_detector.py", "--data", f"{W}/findmine_ft.yaml", "--name", "findmine_runRUNN",
     "--model", base[0], "--epochs", "30", "--lr0", "0.002", "--warmup-epochs", "0", "--close-mosaic", "5"])
''')

export = fill(r'''import json
from ultralytics import YOLO
run = Path("/kaggle/working/runs/detect/findmine_runRUNN")
best = YOLO(run / "weights/best.pt")
out = Path("/kaggle/working/out"); out.mkdir(exist_ok=True)
onnx = Path(best.export(format="onnx", opset=12, imgsz=640, simplify=True, dynamic=False))   # same as run 1 (cv.dnn)
shutil.copy(onnx, out / "findmine12_runRUNN.onnx")
(out / "findmine12_runRUNN.json").write_text(json.dumps([best.names[i] for i in range(len(best.names))]))
shutil.copy(run / "results.csv", out / "results_runRUNN.csv")

# Per-class AP on the fine-tune val set (run-1 val + held-out images of each export), run PREVV vs run RUNN.
rows = ["| class | AP50 run PREVV | AP50 run RUNN |", "|---|---|---|"]
m2 = best.val(data=str(W / "findmine_ft.yaml"), imgsz=640, split="val", plots=False).box
m1 = YOLO(base[0]).val(data=str(W / "findmine_ft.yaml"), imgsz=640, split="val", plots=False).box
ap = lambda m, i: f"{m.ap50[list(m.ap_class_index).index(i)]:.3f}" if i in m.ap_class_index else "-"
for i, name in best.names.items():
    rows.append(f"| {name} | {ap(m1, i)} | {ap(m2, i)} |")
rows.append(f"| **all (mAP50)** | {m1.map50:.3f} | {m2.map50:.3f} |")
(out / "runRUNN_report.md").write_text("# Detector run RUNN (wallet, earbuds, medicine fine-tune)\n\n" + "\n".join(rows) + "\n")
print("\n".join(rows))
''')

nb = {"cells": [md(intro),
                code("!pip install -q ultralytics==8.4.182 onnx==1.23.2 onnxslim==0.1.98"),
                code("%%writefile build_finetune.py\n" + (root / "ml/build_finetune.py").read_text(encoding="utf-8")),
                code("%%writefile train_detector.py\n" + (root / "ml/train_detector.py").read_text(encoding="utf-8")),
                code(setup), code(train), code(export)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(root / f"ml/kaggle_finetune_run{RUN}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("written")
