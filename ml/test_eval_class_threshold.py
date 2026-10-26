"""eval_class_threshold: drops only the chosen class's boxes below the threshold; counts per item.

Run from the repo root:  .venv/Scripts/python ml/test_eval_class_threshold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_class_threshold import measure, strict  # noqa: E402

names = ["keys", "wallet", "phone"]
box = lambda c, p: (c, p, 0, 0, 1, 1)  # noqa: E731
assert strict([box(1, 0.3), box(1, 0.6), box(2, 0.3)], 1, 0.5) == [box(1, 0.6), box(2, 0.3)]
dets = {"wallet__mine": {"plain": [[box(1, 0.3)], [box(1, 0.7)]], "tiles": [[box(1, 0.4)], []], "refs": [[box(1, 0.9)]]},
        "phone__mine": {"plain": [[box(2, 0.3), box(1, 0.4)]], "tiles": [[box(1, 0.6)], [box(1, 0.3)]], "refs": []}}
rates, (refs, room, false) = measure(dets, names, 1, 0.25)
assert rates == {"wallet__mine": (2, 2), "phone__mine": (1, 1)} and refs == (1, 1) and room == (1, 2) and false == (2, 2)
rates, (refs, room, false) = measure(dets, names, 1, 0.5)
assert rates == {"wallet__mine": (1, 2), "phone__mine": (1, 1)} and room == (0, 2) and false == (1, 2)
print("CLASS THRESHOLD TEST PASSED")
