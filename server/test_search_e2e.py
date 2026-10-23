"""End-to-end search over /ws with AWS swapped for an in-memory table and a fake planner (no keys needed).

"My cup" is registered from one COCO photo. Searching that photo -> found, guided, last-seen saved.
Searching a photo with a *different* cup -> never "found".
A dark camera -> planner called -> torch on; every step shows on the /trace page.
Run from the repo root:  .venv/Scripts/python server/test_search_e2e.py
"""
import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2 as cv
import uvicorn
import websockets

import agent.search

import main

ROOT = Path(__file__).resolve().parent.parent
MINE = ROOT / "data/coco/val50/000000002592.jpg"    # detector finds a cup here (0.86)
OTHER = ROOT / "data/coco/val50/000000003845.jpg"   # a different cup (0.83)


class FakeTable:
    def __init__(self):
        self.items = {"cup1": {"item_id": "cup1", "name": "cup", "class": "cup"}}

    def get_item(self, Key):
        return {"Item": self.items[Key["item_id"]]} if Key["item_id"] in self.items else {}

    def update_item(self, Key, UpdateExpression, ExpressionAttributeValues):
        self.items[Key["item_id"]]["last_seen"] = ExpressionAttributeValues[":t"]


img = cv.imread(str(MINE))
_, _, x1, y1, x2, y2 = max(main.detector(img), key=lambda d: d[1])
ref = img[int(y1):int(y2), int(x1):int(x2)]
main.table = FakeTable()
main.plan = lambda obs: ([{"type": "torch", "on": True}], {"planner_event": obs["event"], "model": "fake",
                         "tools": [{"name": "set_torch", "input": {"on": True}}], "latency_ms": 1})
agent.search.COACH_STUCK_SECS = 0.5   # planner after half a second of bad frames instead of 8 s
main.load_refs = lambda item_id: [main.verifier.features(ref), main.verifier.features(cv.flip(ref, 1))]

server = uvicorn.Server(uvicorn.Config(main.app, port=8766, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.1)


async def search(photo, frames=4):
    async with websockets.connect("ws://127.0.0.1:8766/ws") as ws:
        async def send(m):
            await ws.send(m)
            return json.loads(await ws.recv())
        assert (await send(json.dumps({"type": "start", "item_id": "nope"})))["error"] == "item not found"
        started = await send(json.dumps({"type": "start", "item_id": "cup1", "lang": "fr"}))
        assert started["state"] == "SCAN", started
        replies = [await send(photo.read_bytes()) for _ in range(frames)]
        return replies, await send(json.dumps({"type": "answer", "answer": "got_it"})), started["session"]


replies, last, session = asyncio.run(search(MINE))
said = [a.get("key") or a["type"] for r in replies for a in r["actions"]]
print("mine:", [r["state"] for r in replies], said, [d.get("match") for d in replies[-1]["detections"]])
assert "found" in said and replies[-1]["state"] == "GUIDE", replies[-1]
guide = [a for a in replies[-1]["actions"] if a["type"] == "guide"]   # last frame followed by the tracker
assert guide and guide[0]["distance"] in ("near", "arm", "far"), replies[-1]
assert last["state"] == "FOUND" and "save_last_seen" in [a["type"] for a in last["actions"]]
assert "last_seen" in main.table.items["cup1"]

replies, last, _ = asyncio.run(search(OTHER))
said = [a.get("key") or a["type"] for r in replies for a in r["actions"]]
print("other:", [r["state"] for r in replies], said, [d.get("match") for d in replies[-1]["detections"]])
assert "found" not in said and last["state"] != "FOUND"

dark = cv.imencode(".jpg", img // 12)[1].tobytes()


async def dark_room():
    async with websockets.connect("ws://127.0.0.1:8766/ws") as ws:
        await ws.send(json.dumps({"type": "start", "item_id": "cup1"}))
        json.loads(await ws.recv())
        out = []
        for _ in range(8):
            await ws.send(dark)
            out.append(json.loads(await ws.recv()))
            await asyncio.sleep(0.1)
        return out


acts = [a for r in asyncio.run(dark_room()) for a in r["actions"]]
assert {"type": "torch", "on": True} in acts and any(a.get("key") == "fix_dark" for a in acts), acts

page = urllib.request.urlopen(f"http://127.0.0.1:8766/trace/{session}").read().decode()
steps = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8766/trace/{session}.json").read())
assert [r["to"] for r in steps][-1] == "FOUND" and "GUIDE" in page, steps
assert any("set_torch" in json.dumps(r) for rs in main.TRACES.values() for r in rs)
server.should_exit = True
print("E2E SEARCH TEST PASSED")
