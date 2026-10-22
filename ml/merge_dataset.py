"""Merge COCO 2017 + LVIS v1 into a 12-class YOLO dataset and print class/box-size counts.

    ml/.venv/Scripts/python ml/merge_dataset.py            # reads data/coco, data/lvis -> data/findmine

Output (data/findmine/):
    labels/{train2017,val2017}/<id>.txt   YOLO labels
    train.txt, val.txt                     image lists (./images/<split>/<id>.jpg)
    findmine.yaml                          Ultralytics data file
Then put the COCO images at data/findmine/images/{train2017,val2017}/ (unzip train2017.zip / val2017.zip there).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

CLASSES = ["keys", "wallet", "phone", "glasses", "sunglasses", "remote",
           "earbuds", "medicine", "cup", "bottle", "bag", "watch"]
# COCO labels these exhaustively on every image, so COCO boxes win for them.
COCO_MAP = {"cell phone": "phone", "remote": "remote", "cup": "cup", "bottle": "bottle",
            "handbag": "bag", "backpack": "bag"}
# LVIS-only classes (federated labels).
LVIS_MAP = {"key": "keys", "wallet": "wallet", "spectacles": "glasses", "sunglasses": "sunglasses",
            "earphone": "earbuds", "medicine": "medicine", "watch": "watch"}
SIZE_BUCKETS = [(0.001, "<0.1%"), (0.01, "0.1-1%"), (0.1, "1-10%"), (float("inf"), ">=10%")]
MIN_INSTANCES = 1500  # plan target per class


def load(path):
    # Drop segmentation while parsing: LVIS train is 1.1 GB and masks are most of it.
    with open(path, encoding="utf-8") as f:
        return json.load(f, object_hook=lambda d: (d.pop("segmentation", None), d)[1])


def yolo_line(cls, bbox, w, h):
    x, y, bw, bh = bbox
    return f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def merge(cocos, lvises):
    """-> {(split, stem): [(cls, area_frac, line), ...]} for images that have a target box."""
    cls_idx = {n: i for i, n in enumerate(CLASSES)}
    images, coco_boxes = {}, defaultdict(list)
    for d in cocos:
        cat = {c["id"]: cls_idx[COCO_MAP[c["name"]]] for c in d["categories"] if c["name"] in COCO_MAP}
        for a in d["annotations"]:
            if a["category_id"] in cat and not a.get("iscrowd"):
                coco_boxes[a["image_id"]].append((cat[a["category_id"]], a["bbox"]))

    lvis_boxes = defaultdict(list)
    for d in lvises:
        cat = {c["id"]: cls_idx[LVIS_MAP[c["name"]]] for c in d["categories"] if c["name"] in LVIS_MAP}
        for im in d["images"]:
            # An LVIS-only class marked not-exhaustive here means unlabeled instances -> false negatives.
            if not set(im.get("not_exhaustive_category_ids", [])) & cat.keys():
                images[im["id"]] = im
        for a in d["annotations"]:
            if a["category_id"] in cat:
                lvis_boxes[a["image_id"]].append((cat[a["category_id"]], a["bbox"]))

    # ponytail: an LVIS class absent from an image is assumed absent (federated labels can miss some);
    # tighten with neg_category_ids per class if the detector learns to ignore keys/wallets.
    out = {}
    for iid, im in images.items():
        w, h = im["width"], im["height"]
        rows = [(c, bw * bh / (w * h), yolo_line(c, (x, y, bw, bh), w, h))
                for c, (x, y, bw, bh) in coco_boxes[iid] + lvis_boxes[iid] if bw >= 1 and bh >= 1]
        if rows:
            split, name = im["coco_url"].rsplit("/", 2)[-2:]
            out[(split, Path(name).stem)] = rows
    return out


def stats_md(merged):
    inst, imgs = defaultdict(int), defaultdict(set)
    sizes = defaultdict(lambda: defaultdict(int))
    for (split, stem), rows in merged.items():
        for c, frac, _ in rows:
            inst[split, c] += 1
            imgs[split, c].add(stem)
            sizes[c][next(lbl for lim, lbl in SIZE_BUCKETS if frac < lim)] += 1
    head = "| class | train inst | train imgs | val inst | " + " | ".join(l for _, l in SIZE_BUCKETS) + " | flag |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for i, n in enumerate(CLASSES):
        tot = sum(sizes[i].values()) or 1
        pct = " | ".join(f"{100 * sizes[i][l] / tot:.0f}%" for _, l in SIZE_BUCKETS)
        flag = "WEAK" if inst["train2017", i] < MIN_INSTANCES else ""
        lines.append(f"| {n} | {inst['train2017', i]} | {len(imgs['train2017', i])} | "
                     f"{inst['val2017', i]} | {pct} | {flag} |")
    n_tr = sum(s == "train2017" for s, _ in merged)
    lines.append(f"\nImages: {n_tr} train, {len(merged) - n_tr} val. "
                 f"Box size = box area / image area. WEAK = under {MIN_INSTANCES} train instances.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="data/coco/annotations")
    ap.add_argument("--lvis", default="data/lvis")
    ap.add_argument("--out", default="data/findmine")
    ap.add_argument("--stats", default="docs/class_counts.md")
    a = ap.parse_args()

    cocos = [load(f"{a.coco}/instances_{s}2017.json") for s in ("train", "val")]
    lvises = [load(f"{a.lvis}/lvis_v1_{s}.json") for s in ("train", "val")]
    merged = merge(cocos, lvises)

    out = Path(a.out)
    lists = defaultdict(list)
    for (split, stem), rows in sorted(merged.items()):
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(r[2] for r in rows) + "\n")
        lists[split].append(f"./images/{split}/{stem}.jpg")
    for split, name in (("train2017", "train"), ("val2017", "val")):
        (out / f"{name}.txt").write_text("\n".join(lists[split]) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (out / "findmine.yaml").write_text(f"train: train.txt\nval: val.txt\nnames:\n{names}\n")

    md = stats_md(merged)
    print(md)
    Path(a.stats).write_text("# COCO + LVIS class counts (12 classes)\n\n"
                             "Generated by `ml/merge_dataset.py`.\n\n" + md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
