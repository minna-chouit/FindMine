"""SUS scoring + summary table on made-up trials.

Run from the repo root:  .venv/Scripts/python ml/test_summarize_user_tests.py
"""
from summarize_user_tests import summarize, sus_score

assert sus_score([3] * 10) == 50              # all neutral
assert sus_score([5, 1] * 5) == 100           # best possible
assert sus_score([1, 5] * 5) == 0             # worst possible
assert sus_score([4, 2, 4, 2, 4, 2, 4, 2, 4, 2]) == 75

T = lambda s, light, res, time, la="no": {"participant": "P1", "system": s, "light": light, "result": res,  # noqa: E731
                                           "time_s": time, "lookalike": la}
trials = [T("B", "normal", "found", "30"), T("B", "normal", "found", "50", "yes"), T("B", "normal", "timeout", ""),
          T("A", "normal", "wrong_item", "20", "yes"), T("A", "normal", "found", "25"),
          T("A", "low_clutter", "timeout", ""), {"participant": "P2", "system": "B", "light": "normal", "result": "",
                                                  "time_s": "", "lookalike": "no"}]   # not run yet: ignored
sus = [{"participant": "P1", **{f"q{i}": "4" if i % 2 else "2" for i in range(1, 11)}, "helpful": "the beeps",
        "confusing": ""}]
md = summarize(trials, sus)
print(md.encode("ascii", "replace").decode())
assert "| B FindMine | normal | 3 | 67% | 40 s | 0 | 0 of 1 |" in md
assert "| A baseline | normal | 2 | 50% | 25 s | 1 | 1 of 1 |" in md
assert "| A baseline | low_clutter | 1 | 0% | – |" in md
assert "mean 75.0" in md and '"the beeps" (P1)' in md and "1 volunteers, 6 trials" in md
print("SUMMARY TEST PASSED")
