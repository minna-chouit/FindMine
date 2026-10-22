"""End-to-end search over /ws with AWS swapped for an in-memory table (no keys needed).

"My cup" is registered from one COCO photo. Searching that photo -> found, guided, last-seen saved.
Searching a photo with a *different* cup -> never "found".
Run from the repo root:  .venv/Scripts/python server/test_search_e2e.py
"""
import asyncio
import json
import threading
import time
from pathlib import Path

import cv2 as cv
import uvicorn
import websockets

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
        assert (await send(json.dumps({"type": "start", "item_id": "cup1"})))["state"] == "SCAN"
        replies = [await send(photo.read_bytes()) for _ in range(frames)]
        return replies, await send(json.dumps({"type": "answer", "answer": "got_it"}))


replies, last = asyncio.run(search(MINE))
said = [a.get("key") or a["type"] for r in replies for a in r["actions"]]
print("mine:", [r["state"] for r in replies], said, [d.get("match") for d in replies[-1]["detections"]])
assert "found" in said and replies[-1]["state"] == "GUIDE", replies[-1]
assert last["state"] == "FOUND" and "save_last_seen" in [a["type"] for a in last["actions"]]
assert "last_seen" in main.table.items["cup1"]

replies, last = asyncio.run(search(OTHER))
said = [a.get("key") or a["type"] for r in replies for a in r["actions"]]
print("other:", [r["state"] for r in replies], said, [d.get("match") for d in replies[-1]["detections"]])
assert "found" not in said and last["state"] != "FOUND"
server.should_exit = True
print("E2E SEARCH TEST PASSED")
