"""Instance verifier (F5): is this crop *my* item? Three OpenCV cues fused into one confidence.

    embedding  cosine of cv.dnn backbone embeddings (DINOv2-small or MobileNetV3)
    geometry   SIFT + ratio test + cv.findHomography RANSAC inliers (textured items)
    colour     HSV histogram correlation (plain items like mugs)
Fusion = logistic regression (cv.ml) whose weights live in models/verifier_fusion.json,
fitted by `ml/eval_home.py --fit` on FindMine-Home photos (it also stores the found/ask thresholds).
"""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
INLIER_SCALE = 30  # inliers at which geometry counts as fully sure
FUSION = Path(__file__).resolve().parents[2] / "models/verifier_fusion.json"


class Verifier:
    def __init__(self, embed_onnx, fusion=FUSION):
        self.net = cv.dnn.readNetFromONNX(str(embed_onnx))
        self.sift = cv.SIFT_create(nfeatures=500)
        self.bf = cv.BFMatcher(cv.NORM_L2)
        self.w = np.array(json.loads(Path(fusion).read_text())["thetas"]) if Path(fusion).exists() else None

    def features(self, bgr):
        """Everything we need from one crop (reference photo or live candidate)."""
        b = cv.dnn.blobFromImage(cv.resize(bgr, (224, 224), interpolation=cv.INTER_AREA), 1 / 255.0, swapRB=True)
        self.net.setInput((b - MEAN) / STD, "image")
        e = self.net.forward().ravel()
        s = 256 / max(bgr.shape[:2])
        grey = cv.cvtColor(cv.resize(bgr, None, fx=s, fy=s, interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(grey, None)
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        return {"emb": e / np.linalg.norm(e), "pts": np.float32([k.pt for k in kp]), "des": des,
                "hist": cv.normalize(hist, None).ravel()}

    def _inliers(self, q, r):
        if q["des"] is None or r["des"] is None or len(q["des"]) < 2 or len(r["des"]) < 2:
            return 0
        good = [m for m, n in (p for p in self.bf.knnMatch(q["des"], r["des"], k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return 0
        src = q["pts"][[m.queryIdx for m in good]]
        dst = r["pts"][[m.trainIdx for m in good]]
        _, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
        return 0 if mask is None else int(mask.sum())

    def cues(self, q, refs):
        """Best match over the item's reference photos, per cue, each roughly in [0, 1]."""
        return np.array([
            max(float(q["emb"] @ r["emb"]) for r in refs),
            min(max(self._inliers(q, r) for r in refs) / INLIER_SCALE, 1.0),
            max(cv.compareHist(q["hist"], r["hist"], cv.HISTCMP_CORREL) for r in refs),
        ], np.float32)

    def fuse(self, c):
        """Per-cue scores -> confidence in [0, 1]."""
        if self.w is None:  # not calibrated yet: plain average
            return float(np.clip(np.mean(c), 0, 1))
        return float(1 / (1 + np.exp(-(self.w[0] + np.asarray(c) @ self.w[1:]))))

    def __call__(self, q, refs):
        """Confidence in [0, 1] that crop features q are the same item as refs."""
        return self.fuse(self.cues(q, refs))
