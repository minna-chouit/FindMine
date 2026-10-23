"""Guidance maths (F5.4): coarse distance band from apparent size, known class size and camera FOV."""
import math

HFOV = 65  # phone camera horizontal field of view, degrees (calibration knob; also used for clock positions)
# Typical largest side of each class in metres (calibration knobs).
SIZE_M = {"keys": 0.07, "wallet": 0.11, "phone": 0.15, "glasses": 0.14, "sunglasses": 0.14, "remote": 0.18,
          "earbuds": 0.06, "medicine": 0.09, "cup": 0.10, "bottle": 0.24, "bag": 0.35, "watch": 0.05}


def distance(cls, box, img_w):
    """-> ("near" | "arm" | "far", metres). box = (x1, y1, x2, y2) pixels. Pinhole: d = size * f / pixels."""
    f = (img_w / 2) / math.tan(math.radians(HFOV / 2))
    px = max(box[2] - box[0], box[3] - box[1], 1)
    d = SIZE_M.get(cls, 0.15) * f / px
    return ("near" if d < 0.5 else "arm" if d < 1.0 else "far"), round(d, 2)
