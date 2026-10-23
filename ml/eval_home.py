"""Evaluate the real pipeline (coach -> detect -> verify -> decision) on our own photos and videos.

Folder layout (one folder per personal item, class = one of the 12 detector classes):
    data/home/<class>__<name>/ref/*.jpg     3-5 registration photos (item fills the frame, as in the app)
    data/home/<class>__<name>/test/*.jpg    room photos where THIS item is visible
    data/home/<class>__<name>/test/*.mp4    or videos (sampled at --fps frames per second)

For every test frame of item i, each item k of the same class is searched for:
    k == i  -> positive pair (should be found/asked)      k != i -> look-alike pair (must never be "found")

    .venv/Scripts/python ml/eval_home.py [--home data/home] [--fps 2]
Writes eval/home_results.csv (one row per frame x item) and eval/home_summary.md.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from agent.search import ASK, FOUND  # noqa: E402
from pipeline.coach import Coach  # noqa: E402
from pipeline.detect import Detector  # noqa: E402
from pipeline.verify import Verifier  # noqa: E402

IMG = {".jpg", ".jpeg", ".png"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm"}


def frames(folder, fps):
    """(name, BGR image) for every test photo and every sampled video frame."""
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in IMG:
            yield p.name, cv.imread(str(p))
        elif p.suffix.lower() in VID:
            cap = cv.VideoCapture(str(p))
            step = max(1, round((cap.get(cv.CAP_PROP_FPS) or 30) / fps))
            i = 0
            while (ok := cap.grab()):
                if i % step == 0:
                    ok, img = cap.retrieve()
                    if ok:
                        yield f"{p.name}#{i}", img
                i += 1
            cap.release()


def decision(conf):
    return "found" if conf >= FOUND else "ask" if conf >= ASK else "silent"


def evaluate(home, detector, names, verifier, fps=2):
    """-> list of row dicts (one per test frame x same-class item)."""
    if not Path(home).is_dir():
        raise SystemExit(f"{home} not found: create data/home/<class>__<name>/ref and /test (see the top of this file)")
    items = {}
    for d in sorted(p for p in Path(home).iterdir() if p.is_dir() and "__" in p.name):
        cls = d.name.split("__")[0]
        if cls not in names:
            raise SystemExit(f"{d.name}: class '{cls}' is not one of {names}")
        refs = [verifier.features(cv.imread(str(p))) for p in sorted((d / "ref").glob("*")) if p.suffix.lower() in IMG]
        items[d.name] = {"cls": cls, "refs": refs, "dir": d}
    rows = []
    for name, it in items.items():
        same = [k for k, o in items.items() if o["cls"] == it["cls"]]
        coach = Coach()
        for fname, img in frames(it["dir"] / "test", fps):
            if img is None:
                continue
            q = coach(img, now=0.0)  # now=0: every frame judged alone (no shake/blocked history)
            boxes = [b for b in detector(img) if names[b[0]] == it["cls"]]
            feats = [verifier.features(img[int(y1):int(y2), int(x1):int(x2)])
                     for _, _, x1, y1, x2, y2 in boxes if min(y2 - y1, x2 - x1) >= 8]
            for k in same:
                conf = max((verifier(f, items[k]["refs"]) for f in feats), default=0.0) if items[k]["refs"] else 0.0
                rows.append({"item": name, "frame": fname, "searched_for": k, "same_item": int(k == name),
                             "coach_issue": q["issue"] or "", "detected": int(bool(boxes)),
                             "det_conf": round(max((b[1] for b in boxes), default=0.0), 3),
                             "match": round(conf, 3), "decision": decision(conf)})
    return rows


def summary(rows):
    pos = [r for r in rows if r["same_item"]]
    neg = [r for r in rows if not r["same_item"]]
    lines = ["| item | frames | coach flagged | detected | found | asked | silent |", "|---|---|---|---|---|---|---|"]
    by_item = defaultdict(list)
    for r in pos:
        by_item[r["item"]].append(r)
    for name, rs in by_item.items():
        n = len(rs)
        pct = lambda f: f"{100 * sum(map(f, rs)) / n:.0f}%"  # noqa: E731
        lines.append(f"| {name} | {n} | {pct(lambda r: r['coach_issue'] != '')} | {pct(lambda r: r['detected'])} | "
                     f"{pct(lambda r: r['decision'] == 'found')} | {pct(lambda r: r['decision'] == 'ask')} | "
                     f"{pct(lambda r: r['decision'] == 'silent')} |")
    lines += ["", f"| threshold | precision | recall | look-alike frames wrongly at/above it |", "|---|---|---|---|"]
    for label, t in (("found", FOUND), ("ask", ASK)):
        tp = sum(r["match"] >= t for r in pos)
        fp = sum(r["match"] >= t for r in neg)
        lines.append(f"| {label} >= {t} | {tp / max(tp + fp, 1):.2f} | {tp / max(len(pos), 1):.2f} | {fp} of {len(neg)} |")
    lines.append(f"\n{len(pos)} item frames, {len(neg)} look-alike pairs (same class, different item). "
                 "Plan target: precision >= 0.95 at *found* (N3). Coach flags are single-frame (no shake).")
    if not neg:
        lines.append("No look-alikes yet: register two items of the same class to measure wrong-item errors.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(ROOT / "data/home"))
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--model", default=str(ROOT / "models/findmine12.onnx"))
    ap.add_argument("--out", default=str(ROOT / "eval"))
    a = ap.parse_args()
    names = json.loads(Path(a.model).with_suffix(".json").read_text())
    rows = evaluate(a.home, Detector(a.model), names, Verifier(ROOT / "models/dinov2_vits14.onnx"), a.fps)
    if not rows:
        raise SystemExit(f"no test frames found under {a.home}")
    out = Path(a.out)
    with open(out / "home_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = summary(rows)
    print(md)
    (out / "home_summary.md").write_text("# FindMine-Home evaluation\n\nGenerated by `ml/eval_home.py`.\n\n" + md + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
