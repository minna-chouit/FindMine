"""Camera coach (F4): per-frame quality checks with pure OpenCV, one spoken fix per problem.

Runs on a 320 px grey copy. One Coach per connection (shake and blocked-lens need history).
Thresholds are first guesses; tune them on VizWiz quality labels + our bad-camera clips (plan §4).
"""
import time

import cv2 as cv
import numpy as np

WIDTH = 320
BLUR_VAR = 60          # Laplacian variance below this = blurry
DARK_MEAN = 45         # mean grey below this = too dark
DARK_CLIP = 0.6        # or this share of pixels below 10
BRIGHT_CLIP = 0.4      # share of pixels above 245 = overexposed
SHAKE_PX = 25          # median optical-flow move between frames (at 320 px) = shaking, not sweeping
FLAT_GRAD = 8          # Sobel magnitude below this = no texture
BLOCKED_AREA = 0.6     # largest flat region covering this share of the frame...
BLOCKED_MEAN = 90      # ...and darker than this (finger, pocket, case)...
BLOCKED_SECS = 1.0     # ...for this long = lens blocked

FIX = {
    "blocked": {"en": "Something is covering the camera. Move your finger or wipe the lens.",
                "fr": "Quelque chose couvre la caméra. Retirez votre doigt ou essuyez l'objectif.",
                "ar": "شيء ما يغطي الكاميرا. أبعد إصبعك أو امسح العدسة."},
    "dark": {"en": "Too dark. Turn on the torch or move closer to a light.",
             "fr": "Trop sombre. Allumez la lampe ou approchez-vous d'une lumière.",
             "ar": "المكان مظلم جدا. شغّل المصباح أو اقترب من الضوء."},
    "bright": {"en": "Too bright. Turn away from the light.",
               "fr": "Trop lumineux. Détournez-vous de la lumière.",
               "ar": "الضوء قوي جدا. ابتعد عن مصدر الضوء."},
    "shake": {"en": "Slow down.", "fr": "Plus lentement.", "ar": "ببطء."},
    "blur": {"en": "Hold the phone still.", "fr": "Tenez le téléphone immobile.", "ar": "ثبّت الهاتف."},
}


class Coach:
    def __init__(self):
        self.prev = None           # previous grey frame
        self.flat_since = None     # when the dark flat region appeared
        self.last_issue = None

    def __call__(self, bgr, now=None):
        """-> {"issue": code|None, "hint": code|None, "fixed": code|None, "metrics": {...}}.
        issue None = frame is usable; hint = usable, but tell the user how to do better."""
        now = time.monotonic() if now is None else now
        h, w = bgr.shape[:2]
        grey = cv.cvtColor(cv.resize(bgr, (WIDTH, round(h * WIDTH / w)), interpolation=cv.INTER_AREA),
                           cv.COLOR_BGR2GRAY)
        n = grey.size
        hist = cv.calcHist([grey], [0], None, [256], [0, 256]).ravel()
        m = {"blur": float(cv.Laplacian(grey, cv.CV_64F).var()),
             "mean": float(grey.mean()),
             "dark_clip": float(hist[:10].sum() / n),
             "bright_clip": float(hist[246:].sum() / n),
             "shake": self._shake(grey)}

        gx, gy = cv.Sobel(grey, cv.CV_32F, 1, 0), cv.Sobel(grey, cv.CV_32F, 0, 1)
        flat = (cv.magnitude(gx, gy) < FLAT_GRAD).astype(np.uint8)
        k, labels, stats, _ = cv.connectedComponentsWithStats(flat)
        big = 1 + int(stats[1:, cv.CC_STAT_AREA].argmax()) if k > 1 else 0
        m["flat"] = float(stats[big, cv.CC_STAT_AREA] / n) if big else 0.0
        dark_flat = big and m["flat"] > BLOCKED_AREA and grey[labels == big].mean() < BLOCKED_MEAN
        if not dark_flat:
            self.flat_since = None
        elif self.flat_since is None:
            self.flat_since = now

        # issue = frame is useless, skip detection. hint = say the fix but still detect: on our home videos
        # (Oct 24, 5 fps) the detector still found the item in 47% of "blur" frames vs 20% of blurry shake.
        # ponytail: a pitch-black room also reads as "blocked" after 1 s; split them with colour
        # (a finger is red-tinted) if the VizWiz/our clips show confusion.
        issue = hint = None
        if self.flat_since is not None and now - self.flat_since >= BLOCKED_SECS:
            issue = "blocked"
        elif m["mean"] < DARK_MEAN or m["dark_clip"] > DARK_CLIP:
            issue = "dark"
        elif m["bright_clip"] > BRIGHT_CLIP:
            issue = "bright"
        elif m["shake"] > SHAKE_PX:
            if m["blur"] < BLUR_VAR:
                issue = "shake"       # moving and smeared
            else:
                hint = "shake"        # fast sweep but the frame is still sharp
        elif m["blur"] < BLUR_VAR:
            hint = "blur"

        fixed = self.last_issue if issue is None else None   # the agent's "did the fix work?" signal
        self.last_issue = issue
        return {"issue": issue, "hint": hint, "fixed": fixed, "metrics": {k: round(v, 3) for k, v in m.items()}}

    def _shake(self, grey):
        prev, self.prev = self.prev, grey
        if prev is None or prev.shape != grey.shape:
            return 0.0
        pts = cv.goodFeaturesToTrack(prev, maxCorners=100, qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 8:
            return 0.0
        nxt, st, _ = cv.calcOpticalFlowPyrLK(prev, grey, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 8:
            return float(SHAKE_PX + 1)  # features lost between frames = big jump
        return float(np.median(np.linalg.norm((nxt - pts)[ok].reshape(-1, 2), axis=1)))
