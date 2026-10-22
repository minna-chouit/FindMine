# Detector run 1 (Kaggle T4, Oct 21–22)

YOLO11n from COCO weights, 12 classes, imgsz 640, 70 epochs, `ml/train_detector.py` via `ml/kaggle_train_run1.ipynb`.
About 7.4 h in total; the first epoch took 1.8 h (dataset scan + cache), the others about 5 min each. Log: `eval/run1_results.csv`.

| metric (val, 1,341 images) | epoch 1 | epoch 70 |
|---|---|---|
| mAP@50 | 0.116 | **0.280** |
| mAP@50-95 | 0.053 | 0.169 |
| precision | 0.57 | 0.52 |
| recall | 0.14 | 0.29 |

- Plan target: mAP@50 ≥ 0.6 on FindMine-Home. This run is well below it on COCO/LVIS val.
  Note that LVIS labels are not exhaustive, so some correct detections count as false positives here.
- Epochs 2–4 got worse (warm-up learning rate too high for fine-tuning), then improved steadily. mAP was still rising
  slowly at epoch 70.
- Export: `models/findmine12.onnx` (opset 12, 640, 16×8400 output). **cv.dnn parity:** max |cv.dnn − onnxruntime| = 7e-3
  on raw outputs over 50 COCO val images. 64 ms/frame on the laptop CPU, the same as stock YOLO11n.
- Per-class AP is not measured yet (needs the COCO val images: one validation cell on Kaggle).
- Run 2 ideas: top up keys/wallet/earbuds/medicine (Open Images, Roboflow) and FindMine-Home frames; lower `lr0`
  or `warmup_epochs`; `imgsz 800` for small objects; yolo11s if latency allows.
