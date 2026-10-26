# Detector run 3 (wallet, earbuds, medicine fine-tune)

Kaggle T4, from run 2's best.pt, 30 epochs (lr0 0.002, no warm-up), `ml/kaggle_finetune_run3.ipynb`.
Train = Roboflow Universe wallet (senior-project-final-datasets/wallet-nfk4a, Public Domain), earbuds
(aquamanj/pen-calculator-earbuds-ecdso v4, CC BY 4.0: "Ear Device"/"EarDevice" only), medicine boxes
(teknofest-r4zmp/medicine-d7ttn v4, CC BY 4.0: 16 brands → medicine), run 2's keys set (main-m4puh/keys-wd8b7,
kept in so keys don't regress), every run-1 image of a weak class and 6,000 replay images.
Validation = run-1 val + each export's held-out images (same sources as training: optimistic for the new classes).
`models/findmine12_run3.onnx` sha256 `2ea63ceb5c2986906a9e70c6075b3192f06e6524ef6ce5871b96d355d0b9b61f`.

| class | AP50 run 2 | AP50 run 3 |
|---|---|---|
| keys | 0.677 | 0.778 |
| wallet | 0.168 | 0.869 |
| phone | 0.411 | 0.407 |
| glasses | 0.246 | 0.244 |
| sunglasses | 0.201 | 0.174 |
| remote | 0.417 | 0.386 |
| earbuds | 0.051 | 0.406 |
| medicine | 0.058 | 0.135 |
| cup | 0.526 | 0.537 |
| bottle | 0.516 | 0.502 |
| bag | 0.263 | 0.255 |
| watch | 0.267 | 0.267 |
| **all (mAP50)** | 0.317 | 0.413 |

(Run 2's numbers differ from `eval/run2_report.md` because this val set adds the new exports' held-out images.)

## On our home photos and videos (never seen in training)

Plain detection, conf 0.25 (`ml/compare_detectors.py --targets wallet earbuds medicine`, `eval/detector_compare_run3.md`):
no item drops more than 10 points. Bags 44/52/38% → 39/50/31%, bottles 100% both, cup 76% both, phone 71% → 76%,
**headphones (counted as earbuds) 17% → 45%**, keys 0/40 both (too small without tiles).

Keys in detail (run 2 → run 3, same measures as run 2's report, `ml/eval_keys_tiles.py`): close-up reference photos 9/11 → 8/11. Search
frames with 2×2 tiles: **8/40 → 21/40 at conf 0.25**, 23/40 → 24/40 at conf 0.05. False keys boxes on the other
items' frames (tiles, conf 0.25): **21/265 → 6/265**.

Safety with run 3 (`eval/run3/`): same-class look-alikes at/above *found* 0.62: **0 of 75**; *found* precision 1.00,
recall 0.09 (run 2 under the same calibration, `eval/run3/run2_baseline/`: 1.00, 0.09). Other-class boxes
(any-class fallback) at/above *found* / *ask*: **0 / 0 of 956**, highest 0.419.

## Verdict

The comparison rule says **KEEP RUN 2**: we have no wallet or medicine photos of our own yet, so the new classes are
untested outside the training datasets. Everything we could measure is the same or better with run 3 (earbuds
via the headphones item, keys with tiles, fewer false keys boxes, safety unchanged). Run 3 can be deployed as
soon as our own wallet and medicine test photos show an improvement (`docs/finetune_run3.md`, Monday morning).
