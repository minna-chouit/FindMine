"""Verifier precision/recall on a PROXY set until FindMine-Home photos exist.

Each COCO object crop is one "item": 4 synthetic views = reference photos, a 5th view = the live query.
Negatives are queries of *other* items from *other photos* (same photo = shared background):
every item of the same COCO class (look-alikes) plus 5 random items of other classes. Even items fit the cv.ml fusion, odd items are the test.

    .venv/Scripts/python ml/eval_verifier.py [dinov2|mobilenet]
Writes models/verifier_fusion.json and eval/verifier_proxy_<model>.md.
"""
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from pipeline.verify import FUSION, Verifier  # noqa: E402

MODELS = {"dinov2": "dinov2_vits14.onnx", "mobilenet": "mobilenetv3_large.onnx"}
which = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
rng = np.random.default_rng(0)


def view(img, box):
    """A new 'photo' of the object in box: perspective, pad/shift, lighting, sometimes blur."""
    x, y, w, h = box
    pad = rng.uniform(0.0, 0.15, 4) * [w, h, w, h]
    src = np.float32([[x - pad[0], y - pad[1]], [x + w + pad[2], y - pad[1]],
                      [x + w + pad[2], y + h + pad[3]], [x - pad[0], y + h + pad[3]]])
    src += rng.uniform(-0.08, 0.08, (4, 2)).astype(np.float32) * [w, h]
    W, H = round(w * 1.15), round(h * 1.15)
    out = cv.warpPerspective(img, cv.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]])),
                             (W, H), borderMode=cv.BORDER_REPLICATE)
    lut = np.clip(255 * (np.arange(256) / 255) ** rng.uniform(0.6, 1.5), 0, 255).astype(np.uint8)
    out = cv.LUT(out, lut)
    return cv.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5)) if rng.random() < 0.3 else out


coco = json.loads((ROOT / "data/coco/annotations/instances_val2017.json").read_text())
names = {c["id"]: c["name"] for c in coco["categories"]}
files = {p.stem: p for p in (ROOT / "data/coco/val50").glob("*.jpg")}
items = []  # (class name, image, box)
for a in coco["annotations"]:
    stem = f"{a['image_id']:012d}"
    if stem in files and not a["iscrowd"] and min(a["bbox"][2:]) >= 48:
        items.append((names[a["category_id"]], stem, a["bbox"]))
imgs = {s: cv.imread(str(files[s])) for _, s, _ in items}
print(f"{len(items)} items from {len(imgs)} images, embedding = {which}")

v = Verifier(ROOT / "models" / MODELS[which], fusion="none")
refs = [[v.features(view(imgs[s], b)) for _ in range(4)] for _, s, b in items]
queries = [v.features(view(imgs[s], b)) for _, s, b in items]

rows = []  # (item index, query index, label, hard negative?, cues)
for i, (cls, stem, _) in enumerate(items):
    same = [j for j, it in enumerate(items) if it[1] != stem and it[0] == cls]
    other = [j for j, it in enumerate(items) if it[1] != stem and it[0] != cls]
    for j in [i] + same + list(rng.choice(other, min(5, len(other)), replace=False)):
        rows.append((i, j, int(i == j), j in same, v.cues(queries[j], refs[i])))
X = np.array([r[4] for r in rows], np.float32)
y = np.array([r[2] for r in rows], np.float32)
train = np.array([r[0] % 2 == 0 for r in rows])

# Fit the fusion on even items, positives:negatives balanced 1:1.
pos, neg = np.where(train & (y == 1))[0], np.where(train & (y == 0))[0]
fit = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
lr = cv.ml.LogisticRegression_create()
lr.setLearningRate(1.0)
lr.setIterations(20000)
lr.setRegularization(cv.ml.LogisticRegression_REG_DISABLE)
lr.setTrainMethod(cv.ml.LogisticRegression_BATCH)
lr.train(X[fit], cv.ml.ROW_SAMPLE, y[fit].reshape(-1, 1))
thetas = lr.get_learnt_thetas().ravel().tolist()
FUSION.write_text(json.dumps({"thetas": thetas, "cues": ["embedding", "geometry", "colour"],
                              "fitted_on": f"proxy COCO val50, {which}"}, indent=1))


def pr(score, t, mask):
    tp = int(((score >= t) & (y == 1) & mask).sum())
    fp = int(((score >= t) & (y == 0) & mask).sum())
    return tp / max(tp + fp, 1), tp / max(int(((y == 1) & mask).sum()), 1)


test = ~train
fused = 1 / (1 + np.exp(-(thetas[0] + X @ np.array(thetas[1:]))))
hard = np.array([r[3] or r[2] for r in rows])  # positives + look-alike negatives only
lines = [f"# Verifier proxy precision/recall ({which})\n",
         "PROXY: synthetic views of COCO val50 crops, not real items. Replace with FindMine-Home numbers.\n",
         f"{len(items)} items, {int((y[test] == 1).sum())} test positives, "
         f"{int((y[test] == 0).sum())} test negatives ({int((test & hard & (y == 0)).sum())} same-class look-alikes).",
         f"Fusion thetas (bias, emb, geom, colour): {[round(t, 2) for t in thetas]}\n",
         "| score | threshold | precision | recall | precision vs look-alikes only |", "|---|---|---|---|---|"]
for name, s in [("embedding", X[:, 0]), ("geometry", X[:, 1]), ("colour", X[:, 2]), ("fused", fused)]:
    ts = [0.5, 0.8] if name == "fused" else [float(np.median(s[test & (y == 1)]))]
    for t in ts:
        p, r = pr(s, t, test)
        ph, _ = pr(s, t, test & hard)
        lines.append(f"| {name} | {t:.2f} | {p:.2f} | {r:.2f} | {ph:.2f} |")
lines.append("\nSingle cues are thresholded at their median test-positive score (recall about 0.5) for comparison.")
md = "\n".join(lines)
print(md)
(ROOT / "eval").mkdir(exist_ok=True)
(ROOT / f"eval/verifier_proxy_{which}.md").write_text(md + "\n", encoding="utf-8")
