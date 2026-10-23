"""User-test summary on made-up rounds and a made-up Google Form export.

Run from the repo root:  .venv/Scripts/python ml/test_summarize_user_tests.py
"""
from summarize_user_tests import summarize

T = lambda p, s, light, res, time: {"person": p, "system": s, "light": light, "result": res, "time_s": time}  # noqa: E731
trials = [T("1", "B", "normal", "found", "30"), T("1", "B", "normal", "found", "50"), T("2", "B", "normal", "timeout", ""),
          T("1", "A", "normal", "wrong_item", "20"), T("2", "A", "normal", "found", "25"),
          T("2", "A", "low_clutter", "timeout", ""), T("3", "B", "normal", "", "")]      # person 3 not run yet: ignored
form = [{"Timestamp": "x", "Name": "Sara", "The voice was clear": "4", "Best moment?": "the beeps", "Worst?": ""},
        {"Timestamp": "y", "Name": "Amine", "The voice was clear": "5", "Best moment?": "", "Worst?": "too slow"}]
md = summarize(trials, form)
print(md.encode("ascii", "replace").decode())
assert "| B FindMine | normal | 3 | 67% | 40 s | 0 |" in md
assert "| A baseline | normal | 2 | 50% | 25 s | 1 |" in md
assert "| A baseline | low_clutter | 1 | 0% | – | 0 |" in md
assert "2 volunteers, 6 rounds" in md
assert "| The voice was clear | 4.5 |" in md and '- "the beeps"' in md and '- "too slow"' in md
assert "Sara" not in md                                                               # names never in the summary
print("SUMMARY TEST PASSED")
