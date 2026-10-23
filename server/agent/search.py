"""Search state machine (F6.1, F6.4, F6.7): the deterministic real-time path of the agent.

    COACH -> SCAN -> CANDIDATE -> VERIFY -> GUIDE -> FOUND        (sure: conf >= FOUND)
                                      \\-> CONFIRM -> GUIDE ...    (unsure: ask the user first)
    SCAN for too long -> NOT_FOUND

Safety rule: "found" is only spoken when the fused confidence is >= FOUND or the user said yes.
One Search per session. Feed it one observation per frame (step) and the user's answers (user);
both return a list of actions for the phone. Every step is written to the trace callback.
"""
import json
import logging
import time

from pipeline.guide import HFOV
from pipeline.verify import FUSION

_cal = json.loads(FUSION.read_text()) if FUSION.exists() else {}
FOUND = _cal.get("found", 0.8)  # precision >= 95% here (plan N3); set by ml/eval_home.py --fit
ASK = _cal.get("ask", 0.5)      # precision >= 70% here; below it stay silent and keep scanning
VERIFY_FRAMES = 3     # median confidence over this many candidate frames before deciding
CONFIRM_SECS = 10     # no answer to "is it yours?" -> back to scanning
REJECT_SECS = 10      # after "no", don't ask again for this long
LOST_SECS = 3         # target out of view this long while guiding -> back to scanning
STUCK_SECS = 20       # no candidate this long -> planner event (F6.3)
COACH_STUCK_SECS = 8  # bad frames this long -> planner event
GIVE_UP_SECS = 120    # whole search -> NOT_FOUND
REPEAT_SECS = 4       # same spoken coach fix at most this often (F4.2)

PHRASES = {
    "found": {"en": "Found your {item} at {clock} o'clock.", "fr": "{item} trouvé à {clock} heures.",
              "ar": "وجدت {item} على الساعة {clock}."},
    "ask": {"en": "I think I see your {item} at {clock} o'clock. Is it yours?",
            "fr": "Je crois voir votre {item} à {clock} heures. Est-ce le vôtre ?",
            "ar": "أظن أنني أرى {item} على الساعة {clock}. هل هو لك؟"},
    "guide": {"en": "OK, guiding you.", "fr": "D'accord, je vous guide.", "ar": "حسنا، سأرشدك."},
    "keep_looking": {"en": "OK, I'll keep looking.", "fr": "D'accord, je continue à chercher.",
                     "ar": "حسنا، سأواصل البحث."},
    "better": {"en": "Better, continue.", "fr": "C'est mieux, continuez.", "ar": "أفضل، تابع."},
    "lost": {"en": "I lost it. Scan slowly.", "fr": "Je l'ai perdu. Balayez lentement.",
             "ar": "فقدته. حرّك الهاتف ببطء."},
    "turn": {"en": "Turn slowly to look somewhere new.", "fr": "Tournez lentement vers un autre endroit.",
             "ar": "استدر ببطء نحو مكان جديد."},
    "not_found": {"en": "I couldn't find your {item}.{last_seen}", "fr": "Je n'ai pas trouvé {item}.{last_seen}",
                  "ar": "لم أجد {item}.{last_seen}"},
    "done": {"en": "Great. Saved where you found it.", "fr": "Parfait. J'ai noté l'endroit.",
             "ar": "رائع. حفظت المكان."},
}

log = logging.getLogger("findmine.trace")


def clock(cx):
    """Horizontal box centre (0 = left edge, 1 = right edge) -> clock hour, 12 = straight ahead."""
    h = round((cx - 0.5) * HFOV / 30) % 12
    return 12 if h == 0 else h


def say(key, **args):
    return {"type": "speak", "key": key, "text": {lang: t.format(**args) for lang, t in PHRASES[key].items()}}


def log_trace(record):
    log.info(json.dumps(record, ensure_ascii=False))  # JSON lines -> CloudWatch via the container log driver


class Search:
    def __init__(self, item, last_seen=None, now=None, trace=log_trace):
        self.item, self.last_seen, self.trace = item, last_seen, trace
        self.t0 = self.since = self.last_candidate = time.monotonic() if now is None else now
        self.state, self.resume = "SCAN", "SCAN"  # resume = state to go back to after COACH
        self.confs, self.target, self.reject_until = [], None, 0.0
        self.spoken = {}  # coach fix -> last time spoken
        self.stuck_sent = False

    # --- per frame -------------------------------------------------------------------------
    def step(self, quality, candidates=(), now=None):
        """quality: coach output. candidates: [{"box": [x1,y1,x2,y2] normalised 0-1, "conf": fused 0-1}]
        for boxes of the item's class only."""
        now = time.monotonic() if now is None else now
        before, acts, why = self.state, [], ""
        best = max(candidates, key=lambda c: c["conf"], default=None)
        bad = quality.get("issue")

        if self.state in ("FOUND", "NOT_FOUND"):
            return []

        if self.state == "CONFIRM":                       # waiting for yes/no; frames don't matter
            if now - self.since > CONFIRM_SECS:
                why = "no answer"
                self._go("SCAN", now)
        elif self.state == "GUIDE":
            if bad or best is None:                       # shaky hand or out of view: keep the last target
                if now - self.since > LOST_SECS:
                    why = "target lost"
                    acts.append(say("lost"))
                    self._go("SCAN", now)
            else:
                self.since = now
                self.target = best["box"]
                acts.append(self._guide(best))
        elif bad:
            if self.state != "COACH":
                self.resume = self.state if self.state != "VERIFY" else "SCAN"
                self._go("COACH", now)
            if now - self.spoken.get(bad, -1e9) >= REPEAT_SECS:
                self.spoken[bad] = now
                acts.append({"type": "speak", "key": "fix_" + bad, "text": quality.get("fix", {})})
            if now - self.since > COACH_STUCK_SECS:
                acts.append({"type": "planner", "event": "quality_stuck", "issue": bad})
                self.since = now                          # at most one planner call per period
            why = bad
        else:
            if self.state == "COACH":
                why = "quality ok"
                if quality.get("fixed"):
                    acts.append(say("better"))
                self._go(self.resume, now)
            if best is not None and now >= self.reject_until:
                self.last_candidate = now
                if self.state == "SCAN":
                    self._go("CANDIDATE", now)
                self.confs.append(best["conf"])
                self.target = best["box"]
                if len(self.confs) >= VERIFY_FRAMES:
                    why = f"median conf {self._median():.2f}"
                    acts += self._verify(now)
            elif self.state == "CANDIDATE" and now - self.last_candidate > LOST_SECS:
                why = "candidate gone"
                self._go("SCAN", now)
            if self.state == "SCAN":
                if now - self.t0 > GIVE_UP_SECS:
                    why = "time limit"
                    ls = f" Last seen: {self.last_seen}." if self.last_seen else ""
                    acts.append(say("not_found", item=self.item, last_seen=ls))
                    acts.append({"type": "planner", "event": "not_found"})
                    self._go("NOT_FOUND", now)
                elif now - self.last_candidate > STUCK_SECS and not self.stuck_sent:
                    why = "no candidate for 20 s"
                    self.stuck_sent = True
                    acts += [say("turn"), {"type": "planner", "event": "no_candidate"}]

        self._trace(now, before, quality, candidates, why, acts)
        return acts

    # --- user answers ------------------------------------------------------------------------
    def user(self, answer, now=None):
        """answer: "yes" / "no" to "is it yours?", or "got_it" when the item is in hand."""
        now = time.monotonic() if now is None else now
        before, acts = self.state, []
        if self.state == "CONFIRM" and answer == "yes":
            acts.append(say("guide"))
            self._go("GUIDE", now)
        elif self.state == "CONFIRM" and answer == "no":
            acts.append(say("keep_looking"))
            self.reject_until = now + REJECT_SECS
            self._go("SCAN", now)
        elif self.state == "GUIDE" and answer == "got_it":
            acts += [say("done"), {"type": "save_last_seen", "item": self.item}]
            self._go("FOUND", now)
        self._trace(now, before, {}, [], "user: " + answer, acts)
        return acts

    # --- helpers -----------------------------------------------------------------------------
    def _median(self):
        return sorted(self.confs[-VERIFY_FRAMES:])[VERIFY_FRAMES // 2]

    def _verify(self, now):
        self._go("VERIFY", now)
        conf, cx = self._median(), (self.target[0] + self.target[2]) / 2
        if conf >= FOUND:
            self._go("GUIDE", now)
            return [say("found", item=self.item, clock=clock(cx))]
        if conf >= ASK:
            self._go("CONFIRM", now)
            return [say("ask", item=self.item, clock=clock(cx)), {"type": "ask_user", "options": ["yes", "no"]}]
        self._go("SCAN", now)                              # below ASK: stay silent, keep scanning
        return []

    def _guide(self, c):
        x1, y1, x2, y2 = c["box"]
        return {"type": "guide", "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "size": (x2 - x1) * (y2 - y1),
                "clock": clock((x1 + x2) / 2), "distance": c.get("distance")}

    def _go(self, state, now):
        if state == self.state:
            return
        if state == "SCAN":                                # fresh start: forget old evidence, restart timers
            self.confs, self.last_candidate, self.stuck_sent = [], now, False
        self.state, self.since = state, now

    def _trace(self, now, before, quality, candidates, why, acts):
        if before == self.state and not acts:
            return                                        # nothing happened: don't log every frame
        self.trace({"t": round(now - self.t0, 3), "item": self.item, "from": before, "to": self.state,
                    "observation": {"issue": quality.get("issue"),
                                    "candidates": [round(c["conf"], 3) for c in candidates]},
                    "decision": why, "actions": [a.get("key") or a["type"] for a in acts]})
