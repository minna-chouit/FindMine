"""FindMine vision service: phone sends JPEG frames over a WebSocket, gets detections back as JSON.

Run from the repo root:  .venv/Scripts/python -m uvicorn --app-dir server main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import html
import json
import logging
import os
import time
import uuid
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from agent.planner import plan
from agent.search import Search, log_trace
from pipeline.coach import FIX, Coach
from pipeline.guide import distance
from pipeline.track import Track
from pipeline.models import MODEL, detector, names, verifier
from register import load_refs, router as register_router, table

MAX_VERIFY = 2  # verify at most this many boxes per frame (DINOv2 is ~110 ms per crop on a laptop)
TILE_EVERY = 3  # while scanning with no box of the item's class, tile-search every Nth frame (4x detect cost)

log = logging.getLogger("uvicorn.error")
log.info("OpenCV %s | model %s (%d classes)", cv.__version__, MODEL.name, len(names))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(register_router)  # item registration (Takoua): /items


@app.get("/health")
def health():
    return {"ok": True, "opencv": cv.__version__, "model": MODEL.name}


TRACES = {}  # session -> recent trace records, for the /trace page
TRACE_FILE = Path(os.getenv("FINDMINE_TRACE_FILE", Path(__file__).resolve().parent.parent / "data/traces.jsonl"))
TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)


def tracer(session, item_id):
    if len(TRACES) >= 1000:
        TRACES.pop(next(iter(TRACES)))  # oldest session
    records = TRACES[session] = deque(maxlen=1000)

    def write(r):
        r = {**r, "session": session, "item_id": item_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        records.append(r)
        log_trace(r)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:  # survives restarts; data/ is git-ignored
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return write


@app.get("/trace", response_class=HTMLResponse)
def trace_list():
    rows = "".join(f'<li><a href="/trace/{s}">{s}</a> ({len(r)} steps)</li>' for s, r in reversed(TRACES.items()))
    return f'<!doctype html><html lang="en"><title>FindMine traces</title><h1>Search sessions</h1><ul>{rows}</ul></html>'


@app.get("/trace/export.jsonl")
def trace_export():
    """Every trace record written to TRACE_FILE (one JSON per line), for the user-test analysis."""
    return FileResponse(TRACE_FILE, media_type="application/x-ndjson") if TRACE_FILE.exists() else HTMLResponse("", 204)


@app.get("/trace/{session}.json")
def trace_json(session: str):
    return list(TRACES.get(session, []))


@app.get("/trace/{session}", response_class=HTMLResponse)
def trace_page(session: str):
    """Agent trace replay (F6.7): what OpenCV saw, what was decided, what the planner chose."""
    rows = []
    for r in TRACES.get(session, []):
        if "planner_event" in r:
            what = f"planner ({r['planner_event']}, {r['latency_ms']} ms)"
            did = r.get("error") or "; ".join(f"{c['name']}({json.dumps(c['input'], ensure_ascii=False)})" for c in r["tools"])
        else:
            what = f"{r['from']} → {r['to']}"
            obs = r["observation"]
            did = f"picture: {obs['issue'] or 'ok'}; matches: {obs['candidates']} | {r['decision']} | {', '.join(r['actions'])}"
        rows.append(f"<tr><td>{r.get('t', '')}</td><td>{html.escape(what)}</td><td>{html.escape(did)}</td></tr>")
    return ('<!doctype html><html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Trace {html.escape(session)}</title><style>body{{font:16px system-ui;margin:16px;background:#fff;color:#111}}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #888;padding:6px;text-align:left;vertical-align:top}'
            f'</style><h1>Search trace {html.escape(session)}</h1><p><a href="/trace">All sessions</a></p>'
            '<table><thead><tr><th scope="col">time (s)</th><th scope="col">step</th><th scope="col">seen → decided → did</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></html>')


def save_last_seen(item_id):
    # ponytail: time only; add the room label (F7.1) when the app asks the user for it
    table.update_item(Key={"item_id": item_id}, UpdateExpression="SET last_seen = :t",
                      ExpressionAttributeValues={":t": time.strftime("%H:%M on %d %b")})


@app.websocket("/ws")
async def frames(sock: WebSocket):
    """Text messages control a search, binary messages are JPEG frames.

    {"type": "start", "item_id": ..., "lang": "en"|"fr"|"ar"}   start searching for a registered item
    {"type": "answer", "answer": "yes"|"no"|"got_it"}
    {"type": "stop"}
    Every reply has "state" (null when no search) and "actions" (things to speak, ask or guide).
    """
    await sock.accept()
    coach, search, item, refs, target, lang, baseline = Coach(), None, None, [], None, "en", False
    q, best_match, torch, track, n_frames = {}, 0.0, False, None, 0
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
                    lang, best_match = cmd.get("lang", "en"), 0.0
                    baseline = cmd.get("mode") == "baseline"  # user tests, condition A: class only, like other apps
                    search = Search(item["name"], last_seen=item.get("last_seen"), trace=tracer(session, item["item_id"]))
                    reply["session"] = session
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
            n_frames += 1
            dets, match, cues = detector(img), {}, {}
            h, w = img.shape[:2]
            if search:
                mine = [d for d in dets if target is None or d[0] == target]
                if not mine and search.state == "SCAN" and n_frames % TILE_EVERY == 0:
                    mine = [d for d in detector.tiles(img) if target is None or d[0] == target]  # small objects
                    dets = dets + mine
                candidates = []
                if track and search.state == "GUIDE":  # confirmed: follow it, no need to re-verify
                    box = track.update(img, track.match([d[2:] for d in mine]))
                    if box:
                        candidates = [{"box": [box[0] / w, box[1] / h, box[2] / w, box[3] / h], "conf": track.conf,
                                       "distance": distance(track.cls, box, w)[0]}]
                else:
                    for d in sorted(mine, key=lambda d: -d[1])[:MAX_VERIFY]:
                        c, _, x1, y1, x2, y2 = d
                        crop = img[int(y1):int(y2), int(x1):int(x2)]
                        if baseline:      # any box of the class counts as "found" (no instance check)
                            match[id(d)] = 1.0
                        elif refs and min(crop.shape[:2]) >= 8:
                            cues[id(d)] = [round(float(x), 3) for x in verifier.cues(verifier.features(crop), refs)]
                            match[id(d)] = round(verifier.fuse(cues[id(d)]), 3)
                        if id(d) in match:
                            candidates.append({"box": [x1 / w, y1 / h, x2 / w, y2 / h], "conf": match[id(d)],
                                               "px": d[2:], "cls": names[c], "distance": distance(names[c], d[2:], w)[0],
                                               "cues": cues.get(id(d))})
                best_match = max([best_match] + [c["conf"] for c in candidates])
                acts = search.step(q, candidates)
                if search.state != "GUIDE":
                    track = None
                elif track is None and candidates:
                    best = max(candidates, key=lambda c: c["conf"])
                    track = Track(img, best["px"])
                    track.conf, track.cls = best["conf"], best["cls"]
            out = []
            for d in dets:
                c, p, *box = d
                out.append({"label": names[c], "conf": round(p, 3), "box": [round(v) for v in box]}
                           | ({"match": match[id(d)]} if id(d) in match else {}))
            reply = {"quality": q, "w": w, "h": h, "detections": out}
        for ev in [a for a in acts if a["type"] == "planner"]:
            obs = {"event": ev["event"], "lang": lang, "item": item["name"], "state": search.state,
                   "elapsed_s": round(time.monotonic() - search.t0), "last_seen": item.get("last_seen"),
                   "picture": q.get("issue") or "ok", "brightness_0_255": q.get("metrics", {}).get("mean"),
                   "best_match_so_far": round(best_match, 2), "torch_on": torch}
            # ponytail: the phone waits for Bedrock (~1-2 s, rare events); answer asynchronously if it feels laggy
            more, rec = await asyncio.to_thread(plan, obs)
            search.trace({"t": round(time.monotonic() - search.t0, 3), "observation_sent": obs, **rec})
            torch = next((a["on"] for a in more if a["type"] == "torch"), torch)
            acts += more
        if any(a["type"] == "save_last_seen" for a in acts):
            save_last_seen(item["item_id"])
        await sock.send_json(reply | {"state": search and search.state, "actions": acts})
