"""build_finetune.py on a tiny fake run-1 dataset + two fake Roboflow exports (both names: formats).

Run from the repo root:  .venv/Scripts/python ml/test_build_finetune.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_finetune import ALIASES, build, norm, read_names  # noqa: E402

assert ALIASES[norm("Pill_Bottle")] == "medicine" and ALIASES[norm(" AirPods-Case ")] == "earbuds"


def touch(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


with tempfile.TemporaryDirectory() as d:
    root, ex1, ex2 = Path(d) / "findmine", Path(d) / "keys_rf", Path(d) / "ours"
    # run 1: 4 train images (one with keys = weak class), 1 val image
    for i, cls in enumerate([0, 8, 9, 10]):
        touch(root / f"images/train2017/{i}.jpg")
        touch(root / f"labels/train2017/{i}.txt", f"{cls} 0.5 0.5 0.1 0.1\n")
    touch(root / "train.txt", "\n".join(f"./images/train2017/{i}.jpg" for i in range(4)))
    touch(root / "val.txt", "./images/val2017/v.jpg")
    # Roboflow export 1: inline names, class 1 = "Key", class 0 = "Lock" (not ours), polygon line dropped
    touch(ex1 / "data.yaml", "train: ../train/images\nnc: 2\nnames: ['Lock', 'Key']\n")
    for i in range(3):
        touch(ex1 / f"train/images/a{i}.jpg")
        touch(ex1 / f"train/labels/a{i}.txt", "1 0.4 0.4 0.2 0.2\n0 0.1 0.1 0.1 0.1\n1 0.1 0.2 0.3 0.4 0.5 0.6\n")
    touch(ex1 / "train/images/lockonly.jpg")
    touch(ex1 / "train/labels/lockonly.txt", "0 0.5 0.5 0.2 0.2\n")       # no box of ours: skipped
    touch(ex1 / "valid/images/b.jpg")
    touch(ex1 / "valid/labels/b.txt", "1 0.5 0.5 0.3 0.3\n")
    # export 2 (our own frames): block names, no valid split -> 20% held out
    touch(ex2 / "data.yaml", "names:\n- keys\n")
    for i in range(10):
        touch(ex2 / f"train/images/f{i}.jpg")
        touch(ex2 / f"train/labels/f{i}.txt", "0 0.5 0.5 0.1 0.1\n")

    assert read_names(ex1 / "data.yaml") == ["Lock", "Key"] and read_names(ex2 / "data.yaml") == ["keys"]
    train, val = build(root, [ex1, ex2], replay=2)

    assert sum("extra_0" in p for p in train) == 3 and sum("extra_0" in p for p in val) == 1
    assert sum("extra_1" in p for p in train) == 8 and sum("extra_1" in p for p in val) == 2
    assert "./images/train2017/0.jpg" in train                               # weak class always kept
    assert sum("train2017" in p for p in train) == 1 + 2                     # + 2 replay images
    assert "./images/val2017/v.jpg" in val
    lab = (root / "labels/extra_0/train_a0.txt").read_text().split("\n")
    assert lab[0] == "0 0.4 0.4 0.2 0.2" and len([x for x in lab if x]) == 1, lab   # Key -> 0, Lock + polygon gone
    assert all((root / p).exists() for p in train + val if "extra" in p)
    yaml = (root / "findmine_ft.yaml").read_text()
    assert "train: train_ft.txt" in yaml and "0: keys" in yaml and "11: watch" in yaml
print("BUILD FINETUNE TEST PASSED")

# The Kaggle notebook embeds both scripts: they must match the files (else run ml/make_finetune_notebook.py .)
import json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
cells = ["".join(c["source"]) for c in json.load(open(ROOT / "ml/kaggle_finetune_run3.ipynb", encoding="utf-8"))["cells"]]
for script in ("build_finetune.py", "train_detector.py"):
    embedded = next(c for c in cells if c.startswith(f"%%writefile {script}\n"))
    assert embedded.split("\n", 1)[1] == (ROOT / "ml" / script).read_text(encoding="utf-8"), \
        f"notebook copy of {script} is stale: run  .venv/Scripts/python ml/make_finetune_notebook.py ."
print("NOTEBOOK IN SYNC")
