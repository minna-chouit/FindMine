"""Object detection: Ultralytics YOLO exported to ONNX, run with cv.dnn (OpenCV 5)."""
import cv2 as cv
import numpy as np


class Detector:
    def __init__(self, onnx_path, size=640, conf=0.25, iou=0.45):
        self.net = cv.dnn.readNetFromONNX(onnx_path)
        self.size, self.conf, self.iou = size, conf, iou

    def letterbox(self, bgr):
        """Resize keeping aspect ratio, pad to size x size with grey (same as Ultralytics)."""
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        top, left = round((self.size - nh) / 2 - 0.1), round((self.size - nw) / 2 - 0.1)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv.resize(bgr, (nw, nh), interpolation=cv.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, bgr):
        """Return a list of (class_id, confidence, x1, y1, x2, y2) in original image pixels."""
        canvas, r, left, top = self.letterbox(bgr)
        self.net.setInput(cv.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True))
        out = self.net.forward()[0].T  # (anchors, 4 box + num_classes scores)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        boxes, cls, conf = out[keep, :4], cls[keep], conf[keep]
        # cx, cy, w, h on the letterboxed canvas -> x, y, w, h on the original image
        xywh = np.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
        xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
        xywh[:, 2:] = boxes[:, 2:] / r
        idx = cv.dnn.NMSBoxesBatched(xywh.tolist(), conf.tolist(), cls.tolist(), self.conf, self.iou)
        h, w = bgr.shape[:2]
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            dets.append((int(cls[i]), float(conf[i]),
                         float(np.clip(x, 0, w)), float(np.clip(y, 0, h)),
                         float(np.clip(x + bw, 0, w)), float(np.clip(y + bh, 0, h))))
        return dets
