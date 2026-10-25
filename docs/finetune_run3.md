# Detector run 3: wallet, earbuds, medicine (overnight fine-tune)

**Why these three.** They are the weakest classes after run 2 and the ones blind users most need to find and
must not mix up with someone else's:

| class | AP50 run 2 (val) | train images (run 1 data) | why it matters |
|---|---|---|---|
| wallet | 0.014 | 124 | money and cards; lost daily |
| medicine | 0.058 | 39 | own pill box vs a family member's: the look-alike case FindMine exists for |
| earbuds | 0.103 | 576 | small, lost constantly (case on a table, in a bag) |
| glasses / sunglasses | 0.246 / 0.202 | 2,134 / 2,003 | next in line: 4–15× more data already and twice the AP, so less to gain per night |

Keys (0.751) stay in the training set so they don't regress. Run 3 starts from run 2's `best.pt`.

**Rule:** run 2 stays deployed. Run 3 replaces it **only if** `compare_detectors.py --targets wallet earbuds
medicine` says ACCEPT (every target class improves on our photos, no item drops > 10 points, keys included)
**and** the look-alike checks still show 0 at or above *found* (0.62). Production (54.90.98.230) is not touched
by any step here; the swap is a separate decision.

## Tonight, step 1: photograph the test items (30–40 min) — BEFORE training

`data/home` has no wallet, earbuds or medicine items, so without this there is no fair test. These photos are
**never** uploaded to Roboflow or Kaggle: run 3 trains only on public datasets. Don't film these items for
training either.

Six folders, two items per class (yours + a look-alike: a family member's wallet, another pair of earbuds,
another pill box of similar size and colour). The look-alike is what the safety check needs: `eval_home` searches
for item A on item B's photos.

```
data/home/wallet__my_wallet/       data/home/wallet__other_wallet/
data/home/earbuds__my_earbuds/     data/home/earbuds__other_earbuds/
data/home/medicine__my_medicine/   data/home/medicine__other_medicine/
```

In **each** folder:

| subfolder | what | how many |
|---|---|---|
| `ref/` | close-ups, the item fills most of the frame: front, back, side, slightly from above, one in dim light. Plain background. Earbuds: the closed case (that's what's on the table); add one open. Medicine: the box/bottle, label side and back. | 5 photos |
| `test/` | the item **alone** (no other item of its class in view) in a real room at 0.5–3 m: on a desk, bed, floor, shelf, in a bag opening, next to clutter, one dim, one half-hidden. Hold the phone like a search: chest height, slightly down. | 3 photos **or** 2–3 short videos (10–15 s, slow pan) |
| `test/with_other/` | **both** items (yours + the look-alike) in view, any layout, 0.5–2 m | 2 photos or 1 video (only needed in the `my_` folder; fine to copy into both) |

JPG/MP4/MOV (convert HEIC like the other items). Then a quick sanity check that both runs see the new items
(takes ~1 min):

```
.venv/Scripts/python ml/compare_detectors.py --old models/findmine12_run1.onnx --new models/findmine12.onnx --targets wallet earbuds medicine --out eval/home_baseline_check.md
```

It should list the six new items (rates will be low: that's the point). `data/` is git-ignored, so the photos
stay on the laptop only; keep a copy.

## Step 2: Roboflow Universe datasets (you, 15 min)

On universe.roboflow.com search e.g. `wallet`, `earbuds` / `airpods case`, `pill bottle` / `medicine box`.
Pick per class one **Object Detection** project, licence **CC BY 4.0**, real photos (not renders), at least a few
hundred images with boxes. Download Dataset → **YOLOv8** → zip. Note each project's name + licence for the
report credits.

Send me each zip's `data.yaml` `names:` line (or drop the unzipped folders in `data/exports/`). The builder maps
names after lower-casing and reading `_`/`-` as spaces; it already knows wallet/wallets/purse/card holder,
earbuds/earbud/airpods/airpod/earphones/airpods case, medicine/medication/pill/pills/pill bottle/medicine box/
blister pack. Anything else in an export is dropped and **printed as dropped**, so you see it in the Kaggle log.
I'll extend `ALIASES` in `ml/build_finetune.py` if a name is missing (then regenerate the notebook).

Quick local check of an export, without Kaggle:

```
ml/.venv/Scripts/python -c "import sys; sys.path.insert(0,'ml'); from build_finetune import *; n=read_names(sys.argv[1]); print([(x, ALIASES.get(norm(x), 'DROPPED')) for x in n])" PATH/TO/EXPORT/data.yaml
```

## Step 3: start the Kaggle run (10 min, then sleep)

1. kaggle.com → Datasets → New Dataset: upload each new zip as its own dataset.
2. Code → New Notebook → File → Import Notebook → `ml/kaggle_finetune_run3.ipynb`.
3. Add Input:
   - `coco-2017-dataset` (awsaf49)
   - your `findmine-labels` dataset
   - **run 2's notebook** (Notebooks tab; its output holds `runs/detect/findmine_run2/weights/best.pt`)
   - `minnach/keys-computer-vision` (run 2's keys set: keeps keys from regressing)
   - the new wallet / earbuds / medicine datasets
4. Settings: GPU T4 x1 (or P100), Internet on. **Save Version → Save & Run All (Commit)**. Close the browser.

The setup cell prints every input, which `findmine.yaml` it uses (it skips the copy inside run 2's output),
the run-2 weights path and, per export, `name -> class | dropped: [...]`. It **stops** if run 2's weights or the
exports are missing (no silent fallback to stock YOLO). Expected 2–3 h (≈ 11k training images, 30 epochs).

## Monday morning (30 min)

1. Kaggle → the version → Output → `out/findmine12_run3.onnx`, `out/findmine12_run3.json` into `models/`,
   `out/run3_report.md` into `eval/`.
2. On the laptop:

```
.venv/Scripts/python ml/compare_detectors.py --new models/findmine12_run3.onnx --targets wallet earbuds medicine --out eval/detector_compare_run3.md
.venv/Scripts/python ml/eval_home.py --model models/findmine12_run3.onnx --out eval/run3
.venv/Scripts/python ml/eval_cross_class.py --model models/findmine12_run3.onnx
```

   (`eval_cross_class` writes `eval/cross_class.md`; move it to `eval/run3/` and run it once more with the default
   model to restore the run-2 record, same for `eval_home` with `--out eval` if the new items changed it.)
3. **Deploy only if** compare says ACCEPT **and** eval_home shows 0 look-alikes at/above *found* **and**
   eval_cross_class shows 0 at/above *found*. The swap (S3 upload + `infra/deploy.sh`, as in
   `docs/finetune_keys.md`) is your call, given the demo depends on production. Otherwise run 2 stays and run 3
   is a result in the report. `docs/report.md` and `eval/` get updated either way.

## Files

| File | What it does |
|---|---|
| `ml/build_finetune.py` | exports + run-1 weak-class images + 6,000 replay → `findmine_ft.yaml`; prints mapped and dropped names |
| `ml/make_finetune_notebook.py` | writes `ml/kaggle_finetune_run3.ipynb` (embeds the builder and trainer; `ml/test_build_finetune.py` checks they match) |
| `ml/compare_detectors.py` | old vs new on `data/home`, `--targets` classes must improve; tested by `ml/test_compare_detectors.py` |

Tested on the laptop (Oct 25): both tests, and the notebook's own setup → train → export cells run against a fake
`/kaggle` tree (12 COCO images, a stand-in run-2 `best.pt`, a decoy `findmine/` copy, a keys export and a
wallet/medicine export with `Card_Holder`, `Pill-Bottle` and a dropped `coin`), 1 epoch on CPU; the ONNX loads
in `cv.dnn` (OpenCV 5.0.0) through `server/pipeline/detect.py`, and `compare_detectors` correctly rejects it.
