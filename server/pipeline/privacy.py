"""Face blur for every image we store (plan F8.2): OpenCV 5 FaceDetectorYN (YuNet), each face pixelated + blurred.

The score threshold is low on purpose: blurring a bit of wall is fine, leaving a face is not.
"""
import cv2 as cv

SCORE = 0.6     # YuNet's default is 0.9; lower = blur more
PAD = 0.25      # grow each face box by this share on every side (hair, ears, chin)
MAX_SIDE = 640  # detect on a copy at most this big (fast), blur on the full image
PASSES = 3      # detect -> pixelate rounds (each ~10 ms at 640 px)


class FaceBlur:
    def __init__(self, onnx_path):
        self.det = cv.FaceDetectorYN.create(str(onnx_path), "", (320, 320), SCORE, 0.3, 5000)

    def faces(self, bgr):
        """-> face boxes (x1, y1, x2, y2) in bgr's pixels, padded."""
        h, w = bgr.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        small = cv.resize(bgr, (round(w * s), round(h * s))) if s < 1 else bgr
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, found = self.det.detect(small)
        boxes = []
        for x, y, bw, bh in ([] if found is None else found[:, :4] / s):
            px, py = bw * PAD, bh * PAD
            boxes.append((max(0, int(x - px)), max(0, int(y - py)), min(w, int(x + bw + px)), min(h, int(y + bh + py))))
        return boxes

    def __call__(self, bgr):
        """-> (copy of bgr with every face blurred, number of faces blurred).
        Repeats detect -> pixelate: in a crowd, some faces only show up once their neighbours are hidden."""
        out, n = bgr.copy(), 0
        for _ in range(PASSES):
            boxes = self.faces(out)
            if not boxes:
                break
            n += len(boxes)
            for x1, y1, x2, y2 in boxes:
                roi = out[y1:y2, x1:x2]
                if roi.size:  # pixelate to 4x4 blocks, then soften (a plain blur left small faces detectable)
                    blocks = cv.resize(cv.resize(roi, (4, 4), interpolation=cv.INTER_AREA), (x2 - x1, y2 - y1),
                                       interpolation=cv.INTER_NEAREST)
                    out[y1:y2, x1:x2] = cv.GaussianBlur(blocks, (0, 0), max(2, (x2 - x1) / 8))
        return out, n
