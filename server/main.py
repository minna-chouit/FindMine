"""FindMine vision service: phone sends JPEG frames over a WebSocket, gets detections back as JSON.

Run from the repo root:  .venv/Scripts/python -m uvicorn --app-dir server main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pipeline.detect import Detector

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.getenv("FINDMINE_MODEL", ROOT / "models/yolo11n.onnx"))

log = logging.getLogger("uvicorn.error")
detector = Detector(str(MODEL))
names = json.loads(MODEL.with_suffix(".json").read_text())
log.info("OpenCV %s | model %s (%d classes)", cv.__version__, MODEL.name, len(names))

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "model": MODEL.name}


@app.websocket("/ws")
async def frames(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            img = cv.imdecode(np.frombuffer(await sock.receive_bytes(), np.uint8), cv.IMREAD_COLOR)
            if img is None:
                await sock.send_json({"error": "not a JPEG frame"})
                continue
            # ponytail: detection blocks the event loop (~65 ms); fine for one phone, move to a worker thread with its own Net per client if we serve several
            dets = detector(img)
            await sock.send_json({
                "w": img.shape[1], "h": img.shape[0],
                "detections": [{"label": names[c], "conf": round(p, 3), "box": [round(v) for v in box]}
                               for c, p, *box in dets],
            })
    except WebSocketDisconnect:
        pass
