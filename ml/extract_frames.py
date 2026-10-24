"""Videos -> JPEG frames to draw boxes on (Roboflow / CVAT), for the keys fine-tune.

    .venv/Scripts/python ml/extract_frames.py VIDEO [VIDEO ...] --fps 2 --out data/label_keys

Use NEW videos, not the ones in data/home/*/test: those are what run 2 is judged on.
Frames are saved at most 1280 px on the long side; near-identical neighbours (a still phone) are skipped.
"""
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def extract(video, out, fps=2.0, max_side=1280, min_change=4.0):
    """-> number of frames written. min_change = mean abs grey difference to the last kept frame."""
    cap = cv.VideoCapture(str(video))
    step = max(1, round((cap.get(cv.CAP_PROP_FPS) or 30) / fps))
    i = kept = 0
    last = None
    while cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                s = min(1.0, max_side / max(img.shape[:2]))
                img = cv.resize(img, None, fx=s, fy=s, interpolation=cv.INTER_AREA) if s < 1 else img
                small = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
                if last is None or np.abs(small - last).mean() >= min_change:
                    cv.imwrite(str(out / f"{Path(video).stem}_{i:05d}.jpg"), img, [cv.IMWRITE_JPEG_QUALITY, 92])
                    last, kept = small, kept + 1
        i += 1
    cap.release()
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", default="data/label_keys")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(extract(v, out, a.fps) for v in a.videos)
    print(f"{total} frames in {out}  (aim for 40-80 with keys in view; delete frames without keys or add them as negatives)")
