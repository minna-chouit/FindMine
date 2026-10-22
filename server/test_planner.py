"""Planner with a fake Bedrock client: tool calls -> phone actions, errors -> no actions.

Run from the repo root:  .venv/Scripts/python server/test_planner.py
With AWS keys and Bedrock access, `--live` also makes one real call to Claude Haiku 4.5.
"""
import sys

from agent.planner import plan


class FakeBedrock:
    def __init__(self, content=None, error=None):
        self.content, self.error, self.sent = content, error, None

    def converse(self, **kw):
        self.sent = kw
        if self.error:
            raise self.error
        return {"output": {"message": {"role": "assistant", "content": self.content}}}


obs = {"event": "quality_stuck", "lang": "fr", "item": "keys", "picture": "dark", "brightness_0_255": 21,
       "torch_on": False}
fake = FakeBedrock([{"text": "It's dark."},
                    {"toolUse": {"toolUseId": "1", "name": "set_torch", "input": {"on": True}}},
                    {"toolUse": {"toolUseId": "2", "name": "speak", "input": {"text": "J'allume la lampe."}}},
                    {"toolUse": {"toolUseId": "3", "name": "speak", "input": {"text": "third call is dropped"}}}])
acts, rec = plan(obs, bedrock=fake)
assert acts == [{"type": "torch", "on": True},
                {"type": "speak", "key": "planner", "text": {"fr": "J'allume la lampe."}}], acts
assert [t["name"] for t in rec["tools"]] == ["set_torch", "speak"] and "error" not in rec
assert fake.sent["toolConfig"]["toolChoice"] == {"any": {}} and '"lang": "fr"' in fake.sent["messages"][0]["content"][0]["text"]

ask = FakeBedrock([{"toolUse": {"toolUseId": "1", "name": "ask_user",
                                "input": {"question": "Kitchen or bedroom?", "options": ["kitchen", "bedroom"]}}}])
acts, _ = plan({"event": "no_candidate", "lang": "en"}, bedrock=ask)
assert [a["type"] for a in acts] == ["speak", "ask_user"] and acts[1]["options"] == ["kitchen", "bedroom"]

acts, rec = plan(obs, bedrock=FakeBedrock(error=TimeoutError("read timeout")))
assert acts == [] and rec["error"].startswith("TimeoutError"), rec

if "--live" in sys.argv:
    acts, rec = plan(obs)
    print("LIVE:", rec, acts)
    assert acts and "error" not in rec
print("PLANNER TEST PASSED")
