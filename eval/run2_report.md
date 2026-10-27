# Detector run 2 (keys fine-tune)

Kaggle T4, from run 1's best.pt, 30 epochs (lr0 0.002, no warm-up), train = 984 Roboflow Universe keys photos
(main-m4puh/keys-wd8b7, CC BY 4.0) + all run-1 images with a weak class + 6,000 replay images.
Validation = run-1 val + the keys dataset's own held-out images (same source as training: optimistic for keys;
our home videos in eval/detector_compare.md are the fair test).

| class | AP50 run 1 | AP50 run 2 |
|---|---|---|
| keys | 0.004 | 0.751 |
| wallet | 0.011 | 0.014 |
| phone | 0.426 | 0.449 |
| glasses | 0.240 | 0.246 |
| sunglasses | 0.235 | 0.202 |
| remote | 0.445 | 0.419 |
| earbuds | 0.075 | 0.103 |
| medicine | 0.040 | 0.058 |
| cup | 0.571 | 0.535 |
| bottle | 0.553 | 0.517 |
| bag | 0.310 | 0.266 |
| watch | 0.304 | 0.299 |
| **all (mAP50)** | 0.268 | 0.321 |

## On our home photos and videos (never seen in training)

Plain detection, conf 0.25 (`ml/compare_detectors.py`, full table in `eval/detector_compare.md`): keys 0/40 for
both runs; bottles 100% both; cup 82% -> 76%; phone 81% -> 71%; bags -1 frame each; headphones 0% -> 17%.

Keys in detail (run 1 -> run 2): close-up reference photos 0/11 -> **9/11** at conf 0.25 (10/11 at 0.15; `ml/eval_tiles.py`). Search frames: 0/40 -> 2/40 plain,
**23/40 with 2x2 tiles** at conf 0.05. With tiles, keys frames with a keys box: 8/40 at conf 0.25, 17/40 at 0.15;
non-keys frames with a false keys box: 21/265 at 0.25, 26/265 at 0.15.

Safety with run 2: same-class look-alikes at/above *found* 0.62: **0 of 75** (eval_home); other-class boxes
(any-class fallback) at/above *found* / *ask*: **0 / 0 of 939**, highest 0.401 (eval_cross_class).

Reading: run 2 learned keys (it recognises our own keys up close), but in search shots our keys are too small for
the 640 px detector; zoomed tiles find them in about half the frames, mostly below the 0.25 threshold.
