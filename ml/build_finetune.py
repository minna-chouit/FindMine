"""Build a fine-tune dataset: extra YOLO exports (Roboflow Universe sets for weak classes: keys in run 2,
wallet/earbuds/medicine in run 3) on top of a replay sample of the run-1 data, so the detector learns them
without forgetting cups.

    python ml/build_finetune.py --root data/findmine --extra EXPORT_DIR [EXPORT_DIR ...] [--replay 6000]

root   = the run-1 dataset (findmine.yaml, train.txt, val.txt, labels/, images/).
extra  = Roboflow "YOLOv8" exports: data.yaml (names) + {train,valid,test}/{images,labels}/.
Writes into root (Ultralytics finds labels by swapping /images/ for /labels/ in each image path):
    images/extra_<k>/, labels/extra_<k>/   copies of each export, class ids remapped to our 12 classes
    train_ft.txt, val_ft.txt, findmine_ft.yaml
Train: every extra train image + every run-1 image with a weak class (keys, wallet, earbuds, medicine)
+ `replay` random other run-1 images. Val: run-1 val + every extra valid/test image (20% of an export's
train images when it has no valid split).
"""
import argparse
import random
import shutil
from pathlib import Path

CLASSES = ["keys", "wallet", "phone", "glasses", "sunglasses", "remote",
           "earbuds", "medicine", "cup", "bottle", "bag", "watch"]
WEAK = {0, 1, 6, 7}
# Names used by public datasets -> our class. Anything else in an export is dropped.
# Matched after norm(): lower case, "_" and "-" read as spaces.
ALIASES = {"key": "keys", "keys": "keys", "car key": "keys", "house key": "keys",
           "keychain": "keys", "key chain": "keys", "bunch of keys": "keys",
           "wallet": "wallet", "wallets": "wallet", "purse": "wallet", "card holder": "wallet",
           "earbuds": "earbuds", "earbud": "earbuds", "airpods": "earbuds", "airpod": "earbuds",
           "earphones": "earbuds", "earphone": "earbuds", "airpods case": "earbuds", "earbuds case": "earbuds",
           "medicine": "medicine", "medicines": "medicine", "medication": "medicine", "pill bottle": "medicine",
           "pills": "medicine", "pill": "medicine", "medicine box": "medicine", "medicine bottle": "medicine",
           "blister pack": "medicine", "blister": "medicine"}
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(name):
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def read_names(data_yaml):
    """Class names from a Roboflow data.yaml, without a YAML dependency: `names: [a, b]` or a list/dict block."""
    lines = Path(data_yaml).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("names:"):
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("["):
                return [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]
            names = []
            for item in lines[i + 1:]:
                s = item.strip()
                if not s or not item.startswith((" ", "\t", "-")):
                    break
                names.append(s.split(":", 1)[1].strip() if ":" in s and not s.startswith("-") else s.lstrip("- "))
            return [n.strip("'\"") for n in names]
    raise SystemExit(f"no names: in {data_yaml}")


def remap(label_text, id_map):
    """YOLO label text -> lines with our class ids; boxes of unmapped classes are dropped."""
    out = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) == 5 and int(parts[0]) in id_map:   # polygons (more than 5 values) are skipped
            out.append(" ".join([str(id_map[int(parts[0])])] + parts[1:]))
    return out


def import_export(export, root, k, seed=0):
    """Copy one export into root as extra_<k>. -> (train paths, val paths) relative to root."""
    export = Path(export)
    names = read_names(export / "data.yaml")
    id_map = {i: CLASSES.index(ALIASES[norm(n)]) for i, n in enumerate(names) if norm(n) in ALIASES}
    if not id_map:
        raise SystemExit(f"{export}: none of its classes {names} map to ours; add them to ALIASES")
    print(f"{export.name}: " + ", ".join(f"{n} -> {CLASSES[id_map[i]]}" for i, n in id_map.items())
          + f" | dropped: {[n for i, n in enumerate(names) if i not in id_map] or 'none'}")
    splits = {"train": [], "val": []}
    for split, dest in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")):
        for img in sorted((export / split / "images").glob("*")) if (export / split / "images").is_dir() else []:
            if img.suffix.lower() not in IMG:
                continue
            lab = export / split / "labels" / (img.stem + ".txt")
            lines = remap(lab.read_text(encoding="utf-8"), id_map) if lab.exists() else []
            if not lines:
                continue                                  # no box of our classes: skip (not a clean negative)
            name = f"{split}_{img.name}"
            (root / "images" / f"extra_{k}").mkdir(parents=True, exist_ok=True)
            (root / "labels" / f"extra_{k}").mkdir(parents=True, exist_ok=True)
            shutil.copy(img, root / "images" / f"extra_{k}" / name)
            (root / "labels" / f"extra_{k}" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n")
            splits[dest].append(f"./images/extra_{k}/{name}")
    if not splits["val"] and len(splits["train"]) >= 5:   # no valid split: hold out 20% of train
        rnd = random.Random(seed)
        rnd.shuffle(splits["train"])
        n = max(1, len(splits["train"]) // 5)
        splits["val"], splits["train"] = splits["train"][:n], splits["train"][n:]
    print(f"  {len(splits['train'])} train, {len(splits['val'])} val images with our classes")
    return splits["train"], splits["val"]


def has_weak(root, img_path):
    lab = root / img_path.replace("/images/", "/labels/", 1).rsplit(".", 1)[0]
    lab = Path(str(lab) + ".txt")
    return lab.exists() and any(line.split()[0].isdigit() and int(line.split()[0]) in WEAK
                                for line in lab.read_text().splitlines() if line.strip())


def build(root, extras, replay=6000, seed=0):
    root = Path(root)
    base_train = (root / "train.txt").read_text().split()
    base_val = (root / "val.txt").read_text().split()
    train, val = [], list(base_val)
    for k, export in enumerate(extras):
        t, v = import_export(export, root, k, seed)
        train += t
        val += v
    weak = [p for p in base_train if has_weak(root, p)]
    weak_set = set(weak)
    rest = [p for p in base_train if p not in weak_set]
    random.Random(seed).shuffle(rest)
    train += weak + rest[:replay]
    (root / "train_ft.txt").write_text("\n".join(train) + "\n")
    (root / "val_ft.txt").write_text("\n".join(val) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (root / "findmine_ft.yaml").write_text(f"train: train_ft.txt\nval: val_ft.txt\nnames:\n{names}\n")
    print(f"train {len(train)} images (extra {len(train) - len(weak) - min(replay, len(rest))}, "
          f"run-1 weak-class {len(weak)}, replay {min(replay, len(rest))}); val {len(val)}")
    return train, val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/findmine")
    ap.add_argument("--extra", nargs="+", required=True)
    ap.add_argument("--replay", type=int, default=6000)
    a = ap.parse_args()
    build(a.root, a.extra, a.replay)
