# FindMine

Phone web app that helps blind and low-vision people find their own belongings. The vision service runs on OpenCV 5 (`cv.dnn`) on AWS. Plan and requirements: [planAndSRS.md](planAndSRS.md).

## Setup (Windows commands; on Linux/macOS use `python3` and `.venv/bin/`)

Vision server (serving path: OpenCV, no PyTorch):

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Training/export tools (PyTorch, Ultralytics; never used by the server):

```
py -3.12 -m venv ml/.venv
ml/.venv/Scripts/python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
ml/.venv/Scripts/python -m pip install -r ml/requirements.txt
```

## Data and models

Large files stay out of git (see `.gitignore`).

- `data/coco/`: COCO 2017 annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- `data/lvis/`: LVIS v1 annotations: https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip and `lvis_v1_val.json.zip`
- `models/`: run the export spikes to create the ONNX files:

```
ml/.venv/Scripts/python ml/spike_yolo_dnn.py yolo11n
ml/.venv/Scripts/python ml/spike_embed_dnn.py dinov2
```

The first script also downloads 50 COCO val images to `data/coco/val50/`, and the server test uses them. Results are in [docs/spikes.md](docs/spikes.md).

## Run and test the server

```
.venv/Scripts/python -m uvicorn --app-dir server main:app --host 0.0.0.0 --port 8000
.venv/Scripts/python server/test_server.py
.venv/Scripts/python server/test_search_e2e.py   # whole search over /ws, AWS faked in memory
.venv/Scripts/python server/test_planner.py      # add --live on a machine with Bedrock access
.venv/Scripts/python server/test_track.py        # tile search, tracker + Kalman, distance bands
.venv/Scripts/python ml/test_eval_home.py      # evaluation script on a tiny fake home set
```

- `GET /health` returns the OpenCV version and the model name.
- `WS /ws` takes one JPEG per binary message (640 px wide is enough) and replies with one JSON message per frame:
  - usable frame: `{"quality", "w", "h", "detections": [{"label", "conf", "box": [x1, y1, x2, y2]}]}`, with boxes in pixels of the frame you sent
  - bad frame (camera coach): `{"quality": {"issue": "blur|dark|bright|shake|blocked", ...}, "fix": {"en", "fr", "ar"}}`, with no detection
  - not a JPEG: `{"error": "..."}`
- Search (text messages on the same `/ws`):
  - `{"type": "start", "item_id": "..."}` starts searching for a registered item. Frames then also get `"match"` (0–1) on boxes of the item's class. While scanning with no box of that class, every 3rd frame is also searched in 2×2 zoomed tiles (small objects).
  - `{"type": "answer", "answer": "yes" | "no" | "got_it"}`: reply to "is it yours?", or the item is in hand.
  - `{"type": "stop"}`
  - Every reply has `"state"` (`SCAN`, `COACH`, `CANDIDATE`, `CONFIRM`, `GUIDE`, `FOUND`, `NOT_FOUND`, or `null` with no search) and `"actions"`:
    `{"type": "speak", "key", "text": {"en", "fr", "ar"}}`, `{"type": "ask_user", "options": ["yes", "no"]}`,
    `{"type": "guide", "cx", "cy", "size", "clock", "distance": "near"|"arm"|"far"}` (0–1 coordinates, every frame in `GUIDE`, Kalman-smoothed; a CSRT tracker fills frames the detector misses), `{"type": "planner", "event"}`, `{"type": "save_last_seen"}`.
  - "found" is only spoken at match ≥ 0.8 or after the user says yes (`server/agent/search.py`).
- Planner (`server/agent/planner.py`): when the search is stuck (no candidate for 20 s, bad frames for 8 s, not found), the server asks Claude Haiku 4.5 on Bedrock (`FINDMINE_PLANNER_MODEL`, default `us.anthropic.claude-haiku-4-5-20251001-v1:0`, us-east-1) for the next step: `speak`, `set_torch` (phone action `{"type": "torch", "on"}`) or `ask_user`. The server needs `bedrock:InvokeModel` (IAM role on EC2). If Bedrock fails, the fixed phrase is all the user hears. `start` may pass `"lang"` (`en`, `fr`, `ar`) and its reply carries `"session"`.
- `GET /trace/<session>` replays a search: what OpenCV saw, each decision, and the planner's tool calls (`/trace` lists the last 50 sessions, `.json` for raw records).
- `FINDMINE_MODEL` sets the ONNX path (default `models/findmine12.onnx`, our 12 classes; `models/yolo11n.onnx` is the stock 80-class COCO model; the class names come from the `.json` file next to it).

Licences: Ultralytics YOLO is AGPL-3.0. DINOv2 is Apache-2.0.

## Evaluate on our own items

Put each personal item in `data/home/<class>__<name>/` (class = one of the 12 detector classes, e.g. `keys__house_keys`):
`ref/` holds the 3–5 registration photos, `test/` holds room photos or videos where that item is visible.
Two items of the same class (look-alikes) measure wrong-item errors.

```
.venv/Scripts/python ml/eval_home.py
```

It writes `eval/home_results.csv` (one row per frame and item) and `eval/home_summary.md` (per-item detection and decisions, precision/recall at the *found* and *ask* thresholds).

## Deploy on AWS (one command)

`infra/deploy.sh` turns a fresh EC2 instance into the running service: system packages, Python venv with the pinned
requirements, models from S3, a systemd service (starts on boot, restarts on crash), and Caddy for HTTPS on
`<public-ip-with-dashes>.sslip.io`. Re-run it after every `git pull`.

Once per account (AWS console, us-east-1):

1. **Models to S3:** upload `models/findmine12.onnx` and `models/dinov2_vits14.onnx` to `s3://findmine-items-bucket/models/`.
2. **IAM role** for the instance with: `s3:GetObject`/`s3:PutObject` on the bucket, `dynamodb:GetItem`/`PutItem`/`UpdateItem`/`Scan` on `findmine-items`, and `bedrock:InvokeModel`.
3. **Instance:** Ubuntu 24.04 (arm64 for Graviton, e.g. `t4g.medium`/`c7g.large`; x86 `c7i-flex.large` for the baseline), the IAM role above, security group open on 22 (your IP), 80 and 443.

Then on the instance:

```
git clone https://github.com/minna-chouit/FindMine.git && cd FindMine && bash infra/deploy.sh
```

It ends with `DEPLOYED: https://<ip>.sslip.io`. The phone app connects to `wss://<ip>.sslip.io/ws`; the PWA pages are served from `app/`.
Logs: `sudo journalctl -u findmine -f`. Stop the instance when nobody is testing (budget is $0).
If an older copy of the server was started by hand on port 8000, stop it first.
