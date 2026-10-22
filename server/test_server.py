"""Starts the server, sends one COCO image over the WebSocket, checks a person comes back.

Run from the repo root:  .venv/Scripts/python server/test_server.py
"""
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2 as cv
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data/coco/val50/000000000139.jpg"  # living room with two people; fetched by ml/spike_yolo_dnn.py

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "--app-dir", "server", "main:app", "--port", "8765"], cwd=ROOT)
try:
    for _ in range(60):
        try:
            print(urllib.request.urlopen("http://127.0.0.1:8765/health").read().decode())
            break
        except OSError:
            time.sleep(0.5)

    async def check():
        async with websockets.connect("ws://127.0.0.1:8765/ws") as ws:
            await ws.send(IMG.read_bytes())
            reply = json.loads(await ws.recv())
            await ws.send(b"not a jpeg")
            bad = json.loads(await ws.recv())
            dark = cv.imdecode(np.frombuffer(IMG.read_bytes(), np.uint8), cv.IMREAD_COLOR) // 10
            await ws.send(cv.imencode(".jpg", dark)[1].tobytes())
            coached = json.loads(await ws.recv())
        return reply, bad, coached

    reply, bad, coached = asyncio.run(check())
    print(reply)
    assert any(d["label"] == "person" for d in reply["detections"]), "no person detected"
    assert "error" in bad
    assert reply["quality"]["issue"] is None
    assert coached["quality"]["issue"] == "dark" and "detections" not in coached, coached
    print("SERVER TEST PASSED")
finally:
    server.terminate()
