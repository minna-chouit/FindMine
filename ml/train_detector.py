"""Detector training run (GPU box, never AWS). Plan: COCO weights, 12 classes, imgsz 640, 60-80 epochs.

On the GPU machine, from the repo root:
    pip install -r ml/requirements.txt                    # plus a CUDA torch build
    # labels: run ml/merge_dataset.py there, or copy data/findmine/ from a laptop
    mkdir -p data/findmine/images && cd data/findmine/images
    wget http://images.cocodataset.org/zips/train2017.zip http://images.cocodataset.org/zips/val2017.zip
    unzip -q train2017.zip && unzip -q val2017.zip && cd -
    python ml/train_detector.py --epochs 70 --batch -1     # -1 = auto batch for the GPU
Then export for cv.dnn (same settings as ml/spike_yolo_dnn.py):
    yolo export model=runs/detect/findmine_run1/weights/best.pt format=onnx opset=12 imgsz=640 simplify=True
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/findmine/findmine.yaml")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--epochs", type=int, default=70)
ap.add_argument("--batch", type=int, default=-1)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

YOLO(a.model).train(
    data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, name="findmine_run1",
    # Our users: dark rooms, shaky phones, objects cut at the frame edge.
    hsv_v=0.6, degrees=10, translate=0.2, scale=0.6, mosaic=1.0, close_mosaic=10,
    # ponytail: motion blur / low-light gamma+noise come from Ultralytics' built-in albumentations
    # (Blur, MedianBlur, CLAHE) when albumentations is installed; add a custom transform if the
    # FindMine-Home dark/blur split stays weak after run 1.
)
