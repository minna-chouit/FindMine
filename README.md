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
```

- `GET /health` returns the OpenCV version and the model name.
- `WS /ws` takes one JPEG per binary message (640 px wide is enough) and replies with one JSON message per frame:
  - usable frame: `{"quality", "w", "h", "detections": [{"label", "conf", "box": [x1, y1, x2, y2]}]}`, with boxes in pixels of the frame you sent
  - bad frame (camera coach): `{"quality": {"issue": "blur|dark|bright|shake|blocked", ...}, "fix": {"en", "fr", "ar"}}`, with no detection
  - not a JPEG: `{"error": "..."}`
- `FINDMINE_MODEL` sets the ONNX path (default `models/yolo11n.onnx`; the class names come from the `.json` file next to it).

Licences: Ultralytics YOLO is AGPL-3.0. DINOv2 is Apache-2.0.
