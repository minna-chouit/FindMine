"""Day-1 spike: export an embedding backbone to ONNX, run it with cv.dnn, check it matches PyTorch.

Tries DINOv2-small first (plan choice), MobileNetV3-Large as fallback.
Run from the repo root with the ML venv:  ml/.venv/Scripts/python ml/spike_embed_dnn.py [dinov2|mobilenet]
"""
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
which = sys.argv[1] if len(sys.argv) > 1 else "dinov2"
imgs = sorted((ROOT / "data/coco/val50").glob("*.jpg"))[:10]  # fetched by spike_yolo_dnn.py

if which == "dinov2":
    vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

    class ClsEmbedding(torch.nn.Module):
        """Image only in, 384-d CLS embedding out. Calling forward() directly also exports the
        optional `masks` input, and cv.dnn can't run the ops that use it."""
        def __init__(self):
            super().__init__()
            self.vit = vit

        def forward(self, x):
            return self.vit.forward_features(x)["x_norm_clstoken"]

    model = ClsEmbedding()
    name = "dinov2_vits14"
else:
    import torchvision
    m = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2")
    model = torch.nn.Sequential(m.features, m.avgpool, torch.nn.Flatten())  # 960-d global-pooled features
    name = "mobilenetv3_large"
model.eval()

onnx_path = ROOT / f"models/{name}.onnx"
if not onnx_path.exists():
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx_path), opset_version=17,
                      input_names=["image"], output_names=["embedding"], dynamo=False)
print("cv2", cv.__version__, "| onnx", onnx_path)


def blob(path):
    """224x224 centre crop, ImageNet mean/std, RGB, NCHW float32 -- all with OpenCV."""
    img = cv.imread(str(path))
    s = 256 / min(img.shape[:2])
    img = cv.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv.INTER_AREA)
    y, x = (img.shape[0] - 224) // 2, (img.shape[1] - 224) // 2
    img = img[y:y + 224, x:x + 224]
    b = cv.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
    return (b - np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)) / \
        np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


net = cv.dnn.readNetFromONNX(str(onnx_path))
cos, ours, times = [], [], []
for p in imgs:
    b = blob(p)
    net.setInput(b, "image")
    t = time.perf_counter()
    e = net.forward().flatten()
    times.append(time.perf_counter() - t)
    with torch.no_grad():
        r = model(torch.from_numpy(b)).numpy().flatten()
    cos.append(float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r))))
    ours.append(e / np.linalg.norm(e))

# Sanity: a horizontally flipped copy should be closer to its original than other images are
flip = blob(imgs[0])[..., ::-1].copy()
net.setInput(flip, "image")
f = net.forward().flatten()
f /= np.linalg.norm(f)
others = max(float(ours[0] @ o) for o in ours[1:])
print(f"dim {len(ours[0])} | cv.dnn vs PyTorch cosine: min {min(cos):.5f}")
print(f"flip-vs-original {float(ours[0] @ f):.3f}  best other image {others:.3f}")
print(f"cv.dnn median {1000 * float(np.median(times[1:])):.1f} ms/crop (CPU)")
assert min(cos) > 0.999, "cv.dnn output does not match PyTorch"
assert float(ours[0] @ f) > others, "embedding does not rank the same object first"
print("SPIKE PASSED")
