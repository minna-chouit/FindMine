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

Oct 26 we added our own wallet and medicine items (2 wallets, 3 medicine items: a box, a bottle, a blister pack;
reference close-ups, room photos at 1–3 m, look-alikes in view).

Plain detection, conf 0.25 (`ml/compare_detectors.py --targets wallet earbuds medicine`, `eval/detector_compare_run3.md`):
no item drops more than 10 points; **headphones (counted as earbuds) 17% → 45%**; wallets 0/10 both; medicine
1/17 both; keys 0/40 both (too small without tiles). Verdict of the rule: **KEEP RUN 2** (wallet and medicine did
not improve).

With the server's 2×2 tile search and on close-ups (`ml/eval_tiles.py --cls <class>`, run 2 → run 3):

| | close-ups (0.25) | room frames, tiles 0.25 | room frames, tiles 0.05 | false boxes on other items' frames (tiles 0.25) |
|---|---|---|---|---|
| keys | 9/11 → 8/11 | **8/40 → 21/40** | 23/40 → 24/40 | 21/292 → **6/292** |
| wallet | 0/13 → **5/13** | 3/10 → 1/10 | 6/10 → 4/10 | 2/322 → **45/322** |
| medicine | 0/14 → 0/14 | 0/17 → 1/17 | 5/17 → 4/17 | 0/315 → 0/315 |

Safety with run 3, before the wallet/medicine items were added (`eval/run3/`): same-class look-alikes at/above
*found* 0.62: **0 of 75**; *found* precision 1.00, recall 0.09 (run 2 under the same calibration,
`eval/run3/run2_baseline/`: 1.00, 0.09). Other-class boxes at/above *found* / *ask*: **0 / 0 of 956**, highest 0.419.

## Verdict: not deployed, run 2 stays

Run 3 learned wallets up close but not in room shots, and it puts a false "wallet" box on 45 of 322 other frames
(run 2: 2), which would make the app verify and talk about wrong objects more often. The public medicine set
(printed Turkish medicine boxes) did not transfer to our box, bottle and blister pack at all. The gains on
earbuds/headphones and keys are real, but the rule is that the target classes must improve on our own photos.
Next time: wallet and medicine training photos that look like search shots (objects small, in rooms), not
product close-ups; or a separate keys+earbuds-only run.
