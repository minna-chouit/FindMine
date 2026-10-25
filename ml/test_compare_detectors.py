"""compare_detectors.verdict: target classes must improve, nothing may drop > 10 points, no home items = no accept.

Run from the repo root:  .venv/Scripts/python ml/test_compare_detectors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_detectors import verdict  # noqa: E402

old = {"keys__my_keys": (0, 20), "wallet__my_wallet": (2, 20), "headphones__my_headphones": (5, 29), "cup__my_cup": (14, 17)}
ok, why = verdict(old, {**old, "wallet__my_wallet": (9, 20), "headphones__my_headphones": (6, 29)}, ["wallet", "earbuds"])
assert ok, why                                                     # headphones count as earbuds (eval_home ALIASES)
ok, why = verdict(old, {**old, "wallet__my_wallet": (9, 20)}, ["wallet", "earbuds"])
assert not ok and why == ["earbuds did not improve"], why
ok, why = verdict(old, {**old, "wallet__my_wallet": (9, 20), "cup__my_cup": (11, 17)}, ["wallet"])
assert not ok and why[0].startswith("cup__my_cup dropped"), why    # 82% -> 65%
ok, why = verdict({**old, "keys__my_keys": (10, 20)}, {**old, "wallet__my_wallet": (9, 20), "keys__my_keys": (7, 20)}, ["wallet"])
assert not ok and why[0].startswith("keys__my_keys dropped"), why  # old target keys may not regress either
ok, why = verdict(old, old, ["medicine"])
assert not ok and why[0].startswith("no home test items for medicine"), why
assert verdict({"keys__a": (0, 9)}, {"keys__a": (3, 9)})[0]        # run-2 default (keys) still works
print("COMPARE DETECTORS TEST PASSED")
