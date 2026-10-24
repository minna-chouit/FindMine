"""Models loaded once and shared by search (main.py) and registration (register.py)."""
import json
import os
from pathlib import Path

from pipeline.detect import Detector
from pipeline.privacy import FaceBlur
from pipeline.verify import Verifier

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.getenv("FINDMINE_MODEL", ROOT / "models/findmine12.onnx"))
detector = Detector(str(MODEL))
names = json.loads(MODEL.with_suffix(".json").read_text())
verifier = Verifier(os.getenv("FINDMINE_VERIFIER", str(ROOT / "models/dinov2_vits14.onnx")))
face_blur = FaceBlur(os.getenv("FINDMINE_FACE", str(ROOT / "models/face_detection_yunet_2023mar.onnx")))
