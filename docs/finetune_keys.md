# Detector run 2: teach it keys (overnight fine-tune)

**Why:** run 1 gives keys **0 boxes** in all our search-like frames (0 of 35, even at confidence 0.05 and with
zoomed tiles). Bottle, cup, phone and bags are fine. Run 2 starts from run 1's weights and adds keys data,
so it learns keys without forgetting the rest. Training runs on Kaggle (never on AWS).

**Rule:** today's user tests use run 1. Run 2 replaces it **only if** `compare_detectors.py` says ACCEPT on
Monday morning **and** the look-alike checks still pass. Otherwise run 1 stays and run 2 is a result in the report.

## Tonight (about 1 hour of your time, then sleep)

### 1. Film new keys videos (10 min)

Do **not** reuse `data/home/keys__*/test`: those videos are what run 2 is judged on.
Film 4–6 new 15–20 s videos with your phone: keys on a desk, on a bed, on the floor, on a shelf, next to other
things, at 0.5–3 m, normal light and dim light, your keys and your friend's. Copy them to the laptop, then:

```
.venv/Scripts/python ml/extract_frames.py PATH/TO/VIDEO1.MOV PATH/TO/VIDEO2.MOV --out data/label_keys
```

### 2. Draw boxes (30 min)

1. In [Roboflow](https://app.roboflow.com) (free account): New Project → Object Detection → name it `findmine-keys`.
2. Upload `data/label_keys/*.jpg`. Draw one box around each set of keys, class name **`keys`**. Aim for 40–80
   images. Skip blurry frames where you can't see the keys.
3. Generate a version (no extra augmentation, Roboflow's default split is fine) → Export → **YOLOv8** → download zip.

### 3. A public keys dataset (10 min)

Open these on Roboflow Universe, pick the one with the most real photos of keys with **boxes** (task =
Object Detection) and licence **CC BY 4.0**, then Download Dataset → **YOLOv8** → zip:

- https://universe.roboflow.com/keysdataset/full_10k_keys
- https://universe.roboflow.com/ugv-c2f1z/keys-cfm6r
- https://universe.roboflow.com/leonie-xwnrf/key-e2l9b

Write the dataset name and licence in the report (credits). Class names like `Key`, `key`, `car key`, `keychain`
are mapped to our `keys` automatically; anything else in the export is dropped (see `ALIASES` in
`ml/build_finetune.py` to add a name).

### 4. Start the Kaggle run (10 min)

1. kaggle.com → Datasets → New Dataset: upload each zip (your `findmine-keys` export, the Universe export).
2. Code → New Notebook → File → Import Notebook → `ml/kaggle_finetune_run2.ipynb`.
3. Add Input: `coco-2017-dataset` (awsaf49), your `findmine-labels` dataset, **run 1's notebook** (Notebooks tab:
   it holds `findmine_run1/weights/best.pt`) and the two keys datasets.
4. Settings: GPU T4 x1 (or P100), Internet on. **Save Version → Save & Run All (Commit)**. Close the browser.

The first cell prints what it found. If it says "run-1 weights: NOT FOUND", it trains from stock YOLO11n
instead (70 epochs, longer but fine overnight). Expected: about 1.5–3 h with run 1's weights.

## Monday morning (20 min)

1. Kaggle → the notebook's version → Output → download `out/findmine12_run2.onnx`, `out/findmine12_run2.json`
   into `models/` and `out/run2_report.md` (per-class AP, run 1 vs run 2) into `eval/`.
2. On the laptop:

```
.venv/Scripts/python ml/compare_detectors.py --new models/findmine12_run2.onnx
```

3. **ACCEPT?** Then check the look-alike safety with the new detector:

```
.venv/Scripts/python ml/eval_home.py --model models/findmine12_run2.onnx
.venv/Scripts/python ml/eval_cross_class.py --model models/findmine12_run2.onnx
```

   Both must still show **0 look-alikes at or above *found***. Then swap: upload the run-2 ONNX to
   `s3://findmine-items-bucket/models/findmine12.onnx` (replace it; keep a copy of run 1 as `findmine12_run1.onnx`),
   delete `models/findmine12.onnx` on the server and re-run `infra/deploy.sh` (it downloads models that are missing).
   The class list (`models/findmine12.json`, in git) is the same 12 classes, so it doesn't change.

**Done on Oct 25: run 2 is deployed.** Which model is running: `sha256sum models/findmine12.onnx` on the server.

| model | sha256 |
|---|---|
| run 2 (deployed) | `052cfbb7748cf286f266e8583f0553b2fe19bbda7b08ec1362bfd46aec7109ad` |
| run 1 (backup: `s3://findmine-items-bucket/models/findmine12_run1.onnx`) | `7bc0465b848ae7fc7ffb9f51f2cfe99c332592c8f47fafc694170cf911622c9e` |

Roll back: upload `findmine12_run1.onnx` as `models/findmine12.onnx` in S3, delete it on the server, re-run deploy.sh.
   **KEEP RUN 1?** Do nothing to the server; put `eval/detector_compare.md` and `eval/run2_report.md` in the report.

## Files

| File | What it does |
|---|---|
| `ml/extract_frames.py` | videos → frames to label (skips near-identical frames) |
| `ml/build_finetune.py` | keys exports + all run-1 images with weak classes + 6,000 replay images → `findmine_ft.yaml` |
| `ml/train_detector.py` | training (run 2: `--lr0 0.002 --warmup-epochs 0 --close-mosaic 5`, 30 epochs) |
| `ml/kaggle_finetune_run2.ipynb` | the Kaggle notebook (embeds the two scripts above; regenerate with `ml/make_finetune_notebook.py .`) |
| `ml/compare_detectors.py` | run 1 vs run 2 on `data/home`, ACCEPT / KEEP RUN 1 |

Tested on the laptop: the builder (`ml/test_build_finetune.py`) and a 1-epoch CPU run of the full chain
(build → train → export → per-class report → cv.dnn loads the ONNX).
