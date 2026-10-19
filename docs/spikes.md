# Day 1 spikes (Oct 19)

Dev laptop: Windows 11, Python 3.12, CPU only (cv.dnn), OpenCV `5.0.0` (`opencv-contrib-python==5.0.0.93`).
An arm64 wheel exists for the same version (`manylinux2014_aarch64`), so Graviton can use the same pin.

| Spike | Result | Speed (laptop CPU, cv.dnn only) |
|---|---|---|
| YOLO11n (stock COCO), ONNX opset 12, static 640, simplified | **Works.** 278/278 PyTorch boxes matched by cv.dnn (same class, IoU > 0.9) on the first 50 COCO val2017 images, 0 extra boxes | 65 ms/frame incl. letterbox + NMS |
| DINOv2-small (ViT-S/14), ONNX opset 17, 224×224 | **Works**, with a wrapper (see below). cos(cv.dnn, PyTorch) ≥ 0.9993; flipped copy 0.87 vs best other image 0.09 | 110 ms/crop |
| MobileNetV3-Large global-pooled (fallback) | **Works.** cos = 1.0000; flipped copy 0.99 vs best other image 0.38 | 8 ms/crop |

## Gotchas

- **DINOv2 export:** exporting `model.forward` also exports the optional `masks` input, and cv.dnn fails on it (`normalize_axis ... axis is -1`). Export a wrapper that returns `forward_features(x)["x_norm_clstoken"]` instead. Always call `net.setInput(blob, "image")` with the input name.
- **Timing with PyTorch loaded:** if PyTorch is loaded in the same process (as in the parity scripts), it competes with OpenCV for CPU threads, and YOLO looked like 322 ms instead of 65. Time cv.dnn in the server venv only.
- **cv.dnn engines:** OpenCV 5 has `ENGINE_NEW` (default via AUTO), `ENGINE_CLASSIC`, and `ENGINE_ORT`. The pip wheel is built without ONNX Runtime, so ORT falls back to AUTO. YOLO11n forward: NEW 54 ms, CLASSIC 114 ms. Keep the default.

## Open question for Graviton

DINOv2 is 14× slower than MobileNetV3 per crop. We only embed the candidate crops, not every frame, so it may still fit. Measure both on the EC2 instance before choosing.

## Reproduce

```
ml/.venv/Scripts/python ml/spike_yolo_dnn.py yolo11n
ml/.venv/Scripts/python ml/spike_embed_dnn.py dinov2
ml/.venv/Scripts/python ml/spike_embed_dnn.py mobilenet
```
