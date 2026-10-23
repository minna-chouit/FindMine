"""Follow the confirmed item between detections (F3.3): CSRT tracker + Kalman smoothing.

During guidance the detector box (when it matches the track) corrects the filter; when the detector
misses, the CSRT tracker fills in. The Kalman filter (constant velocity on centre and size) keeps the
box, and so the beeps, from jumping.
"""
import cv2 as cv
import numpy as np

MAX_MISSES = 5   # tracker failures in a row before the track is dropped
MIN_IOU = 0.2    # a detection belongs to the track if it overlaps the predicted box this much


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Track:
    def __init__(self, bgr, box):
        """box = (x1, y1, x2, y2) in pixels."""
        self.kf = cv.KalmanFilter(8, 4)  # state cx, cy, w, h + their velocities; measured cx, cy, w, h
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32) + np.eye(8, 8, 4, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1     # calibration knob: how fast it may turn
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 25  # ~5 px measurement noise
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100
        self.kf.statePost = np.r_[self._meas(box), np.zeros((4, 1), np.float32)]
        self.box, self.misses = tuple(map(float, box)), 0
        self._restart(bgr, box)

    @staticmethod
    def _meas(b):
        return np.float32([[(b[0] + b[2]) / 2], [(b[1] + b[3]) / 2], [b[2] - b[0]], [b[3] - b[1]]])

    def _restart(self, bgr, b):
        self.tracker = cv.TrackerCSRT_create()
        self.tracker.init(bgr, (int(b[0]), int(b[1]), max(int(b[2] - b[0]), 1), max(int(b[3] - b[1]), 1)))

    def match(self, boxes):
        """The detection (x1, y1, x2, y2) that continues this track, or None."""
        best = max(boxes, key=lambda b: iou(b, self.box), default=None)
        return best if best is not None and iou(best, self.box) >= MIN_IOU else None

    def update(self, bgr, det_box=None):
        """-> smoothed (x1, y1, x2, y2), or None once the item is lost."""
        self.kf.predict()
        if det_box is not None:
            meas, self.misses = det_box, 0
            self._restart(bgr, det_box)           # re-anchor the tracker on the fresh detection
        else:
            ok, (x, y, w, h) = self.tracker.update(bgr)
            if not ok:
                self.misses += 1
                return None if self.misses > MAX_MISSES else self.box
            meas = (x, y, x + w, y + h)
        cx, cy, w, h = map(float, self.kf.correct(self._meas(meas))[:4, 0])
        self.box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        return self.box
