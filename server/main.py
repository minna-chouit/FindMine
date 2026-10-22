"""FindMine vision service: phone sends JPEG frames over a WebSocket, gets detections back as JSON.

Run from the repo root:  .venv/Scripts/python -m uvicorn --app-dir server main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from agent.search import Search, log_trace
from pipeline.coach import FIX, Coach
from pipeline.detect import Detector
from register import load_refs, router as register_router, table, verifier

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.getenv("FINDMINE_MODEL", ROOT / "models/findmine12.onnx"))
MAX_VERIFY = 2  # verify at most this many boxes per frame (DINOv2 is ~110 ms per crop on a laptop)

log = logging.getLogger("uvicorn.error")
detector = Detector(str(MODEL))
names = json.loads(MODEL.with_suffix(".json").read_text())
log.info("OpenCV %s | model %s (%d classes)", cv.__version__, MODEL.name, len(names))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(register_router)  # item registration (Takoua): /items


@app.get("/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "model": MODEL.name}


def save_last_seen(item_id):
    # ponytail: time only; add the room label (F7.1) when the app asks the user for it
    table.update_item(Key={"item_id": item_id}, UpdateExpression="SET last_seen = :t",
                      ExpressionAttributeValues={":t": time.strftime("%H:%M on %d %b")})


@app.websocket("/ws")
async def frames(sock: WebSocket):
    """Text messages control a search, binary messages are JPEG frames.

    {"type": "start", "item_id": ...}             start searching for a registered item
    {"type": "answer", "answer": "yes"|"no"|"got_it"}
    {"type": "stop"}
    Every reply has "state" (null when no search) and "actions" (things to speak, ask or guide).
    """
    await sock.accept()
    coach, search, item, refs, target = Coach(), None, None, [], None
    session = uuid.uuid4().hex[:8]
    while True:
        msg = await sock.receive()
        if msg["type"] == "websocket.disconnect":
            return
        reply, acts = {}, []
        if msg.get("text") is not None:
            cmd = json.loads(msg["text"])
            if cmd.get("type") == "start":
                item = table.get_item(Key={"item_id": cmd.get("item_id", "")}).get("Item")
                if item:
                    refs = load_refs(item["item_id"])
                    target = names.index(item["class"]) if item.get("class") in names else None  # None = any class
                    search = Search(item["name"], last_seen=item.get("last_seen"), trace=lambda r, i=item["item_id"]:
                                    log_trace({**r, "session": session, "item_id": i}))
                else:
                    reply["error"] = "item not found"
            elif cmd.get("type") == "answer" and search:
                acts = search.user(cmd.get("answer", ""))
            elif cmd.get("type") == "stop":
                search = None
        elif (img := cv.imdecode(np.frombuffer(msg["bytes"], np.uint8), cv.IMREAD_COLOR)) is None:
            reply["error"] = "not a JPEG frame"
        elif (q := coach(img))["issue"]:  # bad frame: speak the fix, skip detection
            reply = {"quality": q, "fix": FIX[q["issue"]]}
            acts = search.step(reply["quality"] | {"fix": reply["fix"]}) if search else []
        else:
            # ponytail: detection + verification block the event loop; fine for one phone, move to a
            # worker thread with its own nets per client if we serve several
            dets = detector(img)
            h, w = img.shape[:2]
            out = [{"label": names[c], "conf": round(p, 3), "box": [round(v) for v in box]} for c, p, *box in dets]
            if search:
                candidates = []
                mine = sorted((d for d in zip(dets, out) if target is None or d[0][0] == target),
                              key=lambda d: -d[0][1])[:MAX_VERIFY]
                for (_, _, x1, y1, x2, y2), o in mine:
                    crop = img[int(y1):int(y2), int(x1):int(x2)]
                    if refs and min(crop.shape[:2]) >= 8:
                        o["match"] = round(verifier(verifier.features(crop), refs), 3)
                        candidates.append({"box": [x1 / w, y1 / h, x2 / w, y2 / h], "conf": o["match"]})
                acts = search.step(q, candidates)
            reply = {"quality": q, "w": w, "h": h, "detections": out}
        if any(a["type"] == "save_last_seen" for a in acts):
            save_last_seen(item["item_id"])
        await sock.send_json(reply | {"state": search and search.state, "actions": acts})
