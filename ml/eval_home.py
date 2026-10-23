"""Evaluate the real pipeline (coach -> detect -> verify -> decision) on our own photos and videos,
and (with --fit) calibrate the verifier's fusion weights and found/ask thresholds on them.

Folder layout (one folder per personal item, class = one of the 12 detector classes):
    data/home/<class>__<name>/ref/*.jpg               3-5 registration photos (item fills the frame)
    data/home/<class>__<name>/test/*.jpg|*.mp4        this item in a room, NO other item of its class in view
    data/home/<class>__<name>/test/with_other/*       this item on the LEFT half, a look-alike on the RIGHT half
Videos are sampled at --fps frames per second.

Rows (one per test frame and searched-for item, best box by fused score):
    alone frames:       searched for its own item -> positive; for each look-alike item -> negative (it isn't there)
    with_other frames:  own item's refs vs left boxes -> positive; vs right boxes (the look-alike) -> negative

    .venv/Scripts/python ml/eval_home.py          evaluate with the current models/verifier_fusion.json
    .venv/Scripts/python ml/eval_home.py --fit    refit fusion + thresholds on these photos, save, re-evaluate
Writes eval/home_results.csv and eval/home_summary.md.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from pipeline.coach import Coach  # noqa: E402
from pipeline.detect import Detector, crop_to_item  # noqa: E402
from pipeline.verify import FUSION, Verifier  # noqa: E402

IMG = {".jpg", ".jpeg", ".png"}
VID = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm"}
FOUND_PRECISION, ASK_PRECISION = 0.95, 0.70   # plan: choose thresholds by precision on our data
MIN_FOUND, MIN_ASK = 0.5, 0.3                 # floors: few look-alikes make low thresholds look "precise"
ALIASES = {"headphones": "earbuds", "earphones": "earbuds", "mug": "cup", "backpack": "bag", "handbag": "bag"}


def thresholds():
    cal = json.loads(FUSION.read_text()) if FUSION.exists() else {}
    return cal.get("found", 0.8), cal.get("ask", 0.5)


def frames(folder, fps):
    """(name, BGR image) for every photo and every sampled video frame directly in folder."""
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in IMG:
            yield p.name, cv.imread(str(p))
        elif p.suffix.lower() in VID:
            cap = cv.VideoCapture(str(p))
            step = max(1, round((cap.get(cv.CAP_PROP_FPS) or 30) / fps))
            i = 0
            while cap.grab():
                if i % step == 0:
                    ok, img = cap.retrieve()
                    if ok:
                        yield f"{p.name}#{i}", img
                i += 1
            cap.release()


def evaluate(home, detector, names, verifier, fps=2, with_other=True):
    """-> list of row dicts with per-cue scores of the best box (by current fusion).
    Also "ref" rows: each reference photo against the item's other references (positive) and against each
    look-alike's references (negative): clean labels, close-up views."""
    if not Path(home).is_dir():
        raise SystemExit(f"{home} not found: create data/home/<class>__<name>/ref and /test (see the top of this file)")
    items = {}
    for d in sorted(p for p in Path(home).iterdir() if p.is_dir() and "__" in p.name):
        cls = d.name.split("__")[0]
        cls = ALIASES.get(cls, cls)
        if cls not in names:
            print(f"skipping {d.name}: '{cls}' is not a detector class ({', '.join(names)})")
            continue
        refs = [verifier.features(crop_to_item(cv.imread(str(p)), detector, names.index(cls)))
                for p in sorted((d / "ref").glob("*")) if p.suffix.lower() in IMG]
        items[d.name] = {"cls": cls, "refs": refs, "dir": d}

    def best(feats, refs):
        """(fused score, cues) of the best-matching box, or zeros."""
        scored = [(verifier.fuse(c), c) for c in (verifier.cues(f, refs) for f in feats)] if refs else []
        return max(scored, key=lambda s: s[0], default=(0.0, np.zeros(3, np.float32)))

    rows = []
    for name, it in items.items():
        others = [k for k, o in items.items() if o["cls"] == it["cls"] and k != name]
        for i, f in enumerate(it["refs"]):
            rest = it["refs"][:i] + it["refs"][i + 1:]
            for k, same, refs in [(name, 1, rest)] + [(k, 0, items[k]["refs"]) for k in others]:
                if refs:
                    c = verifier.cues(f, refs)
                    rows.append({"item": name, "setting": "ref", "frame": f"ref#{i}", "searched_for": k, "same_item": same,
                                 "coach_issue": "", "detected": 1, "det_conf": 1.0, "emb": round(float(c[0]), 4),
                                 "geom": round(float(c[1]), 4), "colour": round(float(c[2]), 4),
                                 "match": round(verifier.fuse(c), 3)})
        for setting in ("alone", "with_other") if with_other else ("alone",):
            folder = it["dir"] / "test" / ("" if setting == "alone" else "with_other")
            for fname, img in frames(folder, fps):
                if img is None:
                    continue
                q = Coach()(img, now=0.0)  # fresh coach: every frame judged alone (no shake/blocked history)
                h, w = img.shape[:2]
                boxes = [b for b in detector(img) if names[b[0]] == it["cls"] and min(b[4] - b[2], b[5] - b[3]) >= 8]
                feats = {id(b): verifier.features(img[int(b[3]):int(b[5]), int(b[2]):int(b[4])]) for b in boxes}
                left = [feats[id(b)] for b in boxes if (b[2] + b[4]) / 2 < w / 2]
                right = [feats[id(b)] for b in boxes if (b[2] + b[4]) / 2 >= w / 2]
                if setting == "alone":
                    pairs = [(name, 1, list(feats.values()))] + [(k, 0, list(feats.values())) for k in others]
                else:
                    pairs = [(name, 1, left), (name, 0, right)]  # right box = the look-alike, searched as mine
                for k, same, fs in pairs:
                    score, c = best(fs, items[k]["refs"])
                    rows.append({"item": name, "setting": setting, "frame": fname, "searched_for": k, "same_item": same,
                                 "coach_issue": q["issue"] or "", "detected": int(bool(fs)),
                                 "det_conf": round(max((b[1] for b in boxes), default=0.0), 3),
                                 "emb": round(float(c[0]), 4), "geom": round(float(c[1]), 4),
                                 "colour": round(float(c[2]), 4), "match": round(score, 3)})
    return rows


def pick_thresholds(scores, labels):
    """Lowest thresholds reaching the target precisions on these rows, never below the floors."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    found = ask = None
    for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
        hit = scores >= t
        if not hit.any():
            break
        precision = labels[hit].mean()
        if ask is None and precision >= ASK_PRECISION and t >= MIN_ASK:
            ask = float(t)
        if found is None and precision >= FOUND_PRECISION and t >= MIN_FOUND:
            found = float(t)
    found = found or 0.95
    return found, min(ask or found, found)


def fit(rows):
    """Refit the 3-cue logistic regression (cv.ml) on detected rows and pick thresholds; save to FUSION."""
    use = [r for r in rows if r["detected"]]
    X = np.float32([[r["emb"], r["geom"], r["colour"]] for r in use])
    y = np.float32([r["same_item"] for r in use])
    if y.sum() < 5 or (1 - y).sum() < 5:
        raise SystemExit(f"need >= 5 positive and >= 5 look-alike rows with a detection, have {int(y.sum())} / "
                         f"{int((1 - y).sum())}: add look-alike items (same class) and their photos")
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    idx = np.concatenate([big, np.resize(small, len(big))])  # balance classes by repeating the smaller one
    lr = cv.ml.LogisticRegression_create()
    lr.setLearningRate(1.0)
    lr.setIterations(20000)
    lr.setRegularization(cv.ml.LogisticRegression_REG_DISABLE)
    lr.setTrainMethod(cv.ml.LogisticRegression_BATCH)
    lr.train(X[idx], cv.ml.ROW_SAMPLE, y[idx].reshape(-1, 1))
    thetas = lr.get_learnt_thetas().ravel()
    scores = 1 / (1 + np.exp(-(thetas[0] + X @ thetas[1:])))
    found, ask = pick_thresholds(scores, y)
    FUSION.write_text(json.dumps({
        "thetas": [float(t) for t in thetas], "cues": ["embedding", "geometry", "colour"],
        "found": found, "ask": ask,
        "fitted_on": f"FindMine-Home {time.strftime('%Y-%m-%d')}: {int(y.sum())} positive, {int((1 - y).sum())} look-alike rows",
    }, indent=1))
    return thetas, found, ask


def summary(rows, found, ask):
    decision = lambda m: "found" if m >= found else "ask" if m >= ask else "silent"  # noqa: E731
    pos = [r for r in rows if r["same_item"]]
    neg = [r for r in rows if not r["same_item"]]
    lines = ["| item | setting | frames | coach flagged | detected | found | asked | silent |",
             "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in pos:
        by[r["item"], r["setting"]].append(r)
    for (name, setting), rs in sorted(by.items()):
        n = len(rs)
        pct = lambda f: f"{100 * sum(map(f, rs)) / n:.0f}%"  # noqa: E731
        lines.append(f"| {name} | {setting} | {n} | {pct(lambda r: r['coach_issue'] != '')} | {pct(lambda r: r['detected'])} | "
                     f"{pct(lambda r: decision(r['match']) == 'found')} | {pct(lambda r: decision(r['match']) == 'ask')} | "
                     f"{pct(lambda r: decision(r['match']) == 'silent')} |")
    lines += ["", "| threshold | precision | recall | look-alike rows wrongly at/above it |", "|---|---|---|---|"]
    for label, t in (("found", found), ("ask", ask)):
        tp = sum(r["match"] >= t for r in pos)
        fp = sum(r["match"] >= t for r in neg)
        lines.append(f"| {label} >= {t:.2f} | {tp / max(tp + fp, 1):.2f} | {tp / max(len(pos), 1):.2f} | {fp} of {len(neg)} |")
    mean = lambda rs, k: np.mean([r[k] for r in rs if r["detected"]]) if any(r["detected"] for r in rs) else 0  # noqa: E731
    lines += ["", "| mean cue score (detected rows) | embedding | geometry | colour |", "|---|---|---|---|",
              f"| own item | {mean(pos, 'emb'):.2f} | {mean(pos, 'geom'):.2f} | {mean(pos, 'colour'):.2f} |",
              f"| look-alike | {mean(neg, 'emb'):.2f} | {mean(neg, 'geom'):.2f} | {mean(neg, 'colour'):.2f} |"]
    lines.append(f"\n{len(pos)} own-item rows, {len(neg)} look-alike rows. Plan target: precision >= 0.95 at *found* (N3). "
                 "Coach flags are single-frame (no shake).")
    if not neg:
        lines.append("No look-alikes yet: add two items of the same class to measure wrong-item errors.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(ROOT / "data/home"))
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--model", default=str(ROOT / "models/findmine12.onnx"))
    ap.add_argument("--out", default=str(ROOT / "eval"))
    ap.add_argument("--fit", action="store_true", help="refit fusion weights + thresholds on these photos")
    ap.add_argument("--no-with-other", action="store_true", help="ignore test/with_other (e.g. left/right rule not kept)")
    a = ap.parse_args()
    names = json.loads(Path(a.model).with_suffix(".json").read_text())
    verifier = Verifier(ROOT / "models/dinov2_vits14.onnx")
    rows = evaluate(a.home, Detector(a.model), names, verifier, a.fps, with_other=not a.no_with_other)
    if not rows:
        raise SystemExit(f"no test frames found under {a.home}")
    note = ""
    if a.fit:
        thetas, found, ask = fit(rows)
        verifier.w = thetas
        for r in rows:  # rescore with the new weights (same cues)
            r["match"] = round(verifier.fuse([r["emb"], r["geom"], r["colour"]]), 3) if r["detected"] else 0.0
        note = (f"Fusion refit on these rows (in-sample, so optimistic): thetas {[round(float(t), 2) for t in thetas]}, "
                f"found {found:.2f}, ask {ask:.2f}. Saved to models/verifier_fusion.json.\n\n")
    found, ask = thresholds()
    with open(Path(a.out) / "home_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = note + summary(rows, found, ask)
    print(md.encode("ascii", "replace").decode())
    (Path(a.out) / "home_summary.md").write_text("# FindMine-Home evaluation\n\nGenerated by `ml/eval_home.py`.\n\n" + md
                                                 + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
