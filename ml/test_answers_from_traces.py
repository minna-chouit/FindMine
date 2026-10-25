"""answers_from_traces: asks paired with yes / no / timeout, per session; the "move closer" step is not an ask.

Run from the repo root:  .venv/Scripts/python ml/test_answers_from_traces.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answers_from_traces import pairs, summary  # noqa: E402


def rec(s, t, decision, actions=()):
    return {"session": s, "t": t, "decision": decision, "actions": list(actions), "from": "X", "to": "Y"}


recs = [
    rec("a", 1.0, "median conf 0.50", ["closer"]),                 # far: not a question
    rec("a", 3.0, "median conf 0.52", ["ask", "ask_user"]),
    rec("a", 4.0, "user: yes", ["guide"]),
    rec("a", 9.0, "user: got_it", ["done", "save_last_seen"]),     # not an answer to "is it yours?"
    rec("b", 2.0, "median conf 0.45", ["ask", "ask_user"]),
    rec("b", 3.0, "user: no", ["keep_looking"]),
    rec("b", 20.0, "median conf 0.60", ["ask", "ask_user"]),
    rec("b", 31.0, "no answer"),
    {"session": "b", "t": 5.0, "planner_event": "no_candidate", "tools": []},   # planner record: ignored
]
ps = sorted(pairs(recs))
assert ps == [("a", 0.52, "yes"), ("b", 0.45, "no"), ("b", 0.6, "none")], ps
md = summary(ps)
assert "Asked 3 times in 2 searches: 1 yes, 1 no, 1 no answer." in md and "0.50" in md, md
print("ANSWERS TEST PASSED")
