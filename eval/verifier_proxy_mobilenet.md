# Verifier proxy precision/recall (mobilenet)

PROXY: synthetic views of COCO val50 crops, not real items. Replace with FindMine-Home numbers.

187 items, 93 test positives, 3623 test negatives (3158 same-class look-alikes).
Fusion thetas (bias, emb, geom, colour): [-3.95, 1.91, 3.97, 2.15]

| score | threshold | precision | recall | precision vs look-alikes only |
|---|---|---|---|---|
| embedding | 0.93 | 1.00 | 0.51 | 1.00 |
| geometry | 1.00 | 1.00 | 0.82 | 1.00 |
| colour | 0.95 | 0.24 | 0.51 | 0.25 |
| fused | 0.50 | 0.98 | 0.97 | 0.98 |
| fused | 0.80 | 1.00 | 0.92 | 1.00 |

Single cues are thresholded at their median test-positive score (recall about 0.5) for comparison.
