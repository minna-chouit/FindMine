"""Search state machine: every path + the safety rule (never "found" below FOUND without a yes).

Run from the repo root:  .venv/Scripts/python server/test_search.py
"""
import random

from agent.search import FOUND, Search, clock

OK = {"issue": None}
DARK = {"issue": "dark", "fix": {"en": "Too dark."}}
BOX = [0.7, 0.4, 0.9, 0.6]   # right of centre -> 1 o'clock


def keys(acts):
    return [a.get("key") or a["type"] for a in acts]


def new(**kw):
    trace = []
    return Search("keys", now=0.0, trace=trace.append, **kw), trace


assert (clock(0.5), clock(1.0), clock(0.0)) == (12, 1, 11)

# Sure match: 3 frames >= FOUND -> "found" + guide -> user has it -> FOUND, save last-seen.
s, trace = new()
for t in (0.2, 0.4):
    assert s.step(OK, [{"box": BOX, "conf": 0.9}], now=t) == [] and s.state == "CANDIDATE"
acts = s.step(OK, [{"box": BOX, "conf": 0.9}], now=0.6)
assert keys(acts) == ["found"] and s.state == "GUIDE" and "1 o'clock" in acts[0]["text"]["en"], acts
assert keys(s.step(OK, [{"box": BOX, "conf": 0.9}], now=0.8)) == ["guide"]
assert keys(s.user("got_it", now=1.0)) == ["done", "save_last_seen"] and s.state == "FOUND"
assert [r["to"] for r in trace] == ["CANDIDATE", "GUIDE", "GUIDE", "FOUND"], trace

# Unsure match: ask, user says no -> silent for REJECT_SECS -> asks again later.
s, _ = new()
for t in (0.2, 0.4, 0.6):
    acts = s.step(OK, [{"box": BOX, "conf": 0.6}], now=t)
assert keys(acts) == ["ask", "ask_user"] and s.state == "CONFIRM"
assert keys(s.user("no", now=1.0)) == ["keep_looking"] and s.state == "SCAN"
assert s.step(OK, [{"box": BOX, "conf": 0.6}], now=5.0) == [] and s.state == "SCAN"   # rejected
for t in (11.2, 11.4, 11.6):
    acts = s.step(OK, [{"box": BOX, "conf": 0.6}], now=t)
assert keys(acts) == ["ask", "ask_user"]
assert keys(s.user("yes", now=12.0)) == ["guide"] and s.state == "GUIDE"
# ...then the target leaves the view for > LOST_SECS -> "lost", back to scanning.
s.step(OK, [], now=13.0)
assert keys(s.step(OK, [], now=15.5)) == ["lost"] and s.state == "SCAN"

# No answer to "is it yours?" -> back to scanning.
s, _ = new()
for t in (0.2, 0.4, 0.6):
    s.step(OK, [{"box": BOX, "conf": 0.6}], now=t)
s.step(OK, [], now=11.0)
assert s.state == "SCAN"

# Weak match: silent, back to scanning.
s, _ = new()
for t in (0.2, 0.4, 0.6):
    acts = s.step(OK, [{"box": BOX, "conf": 0.3}], now=t)
assert acts == [] and s.state == "SCAN"

# Coach: fix spoken at most every 4 s, planner after 8 s of bad frames, "better" when fixed.
s, _ = new()
out = {t: keys(s.step(DARK, now=float(t))) for t in range(0, 11)}
assert [t for t in out if "fix_dark" in out[t]] == [0, 4, 8] and s.state == "COACH", out
assert [t for t in out if "planner" in out[t]] == [9], out   # first frame after 8 s, then not again
assert keys(s.step({"issue": None, "fixed": "dark"}, now=11.0)) == ["better"] and s.state == "SCAN"

# Nothing seen: planner nudge at 20 s (once), give up at 120 s with last-seen.
s, _ = new(last_seen="kitchen, 14:02")
assert keys(s.step(OK, now=21.0)) == ["turn", "planner"]
assert s.step(OK, now=30.0) == []
acts = s.step(OK, now=121.0)
assert keys(acts) == ["not_found", "planner"] and s.state == "NOT_FOUND"
assert "kitchen, 14:02" in acts[0]["text"]["en"]

# A weak box of the class keeps appearing (bottle seen, never close to a match): the user must not be left in
# silence. Seen on the real server (session 0b36fbe0): SCAN <-> CANDIDATE every ~1.2 s for 25 s, nothing said.
s, _ = new()
events = []
for i in range(1, 251):                     # 50 s at 5 fps
    acts = s.step(OK, [{"box": BOX, "conf": 0.22}], now=i * 0.2)
    events += [i * 0.2 for a in acts if a["type"] == "planner"]
assert len(events) == 2 and 20 < events[0] < 21 and 40 < events[1] < 42, events

# Safety fuzz: confidences below FOUND and random "no"/silence never produce "found".
rng = random.Random(0)
for _ in range(200):
    s, _ = new()
    t = 0.0
    for _ in range(300):
        t += 0.2
        q = DARK if rng.random() < 0.1 else OK
        c = [{"box": BOX, "conf": rng.uniform(0, FOUND - 1e-6)}] if rng.random() < 0.7 else []
        acts = s.step(q, c, now=t)
        if s.state == "CONFIRM" and rng.random() < 0.5:
            acts += s.user("no", now=t)
        assert "found" not in keys(acts) and s.state != "GUIDE", (t, s.state, acts)
print("SEARCH TEST PASSED")
