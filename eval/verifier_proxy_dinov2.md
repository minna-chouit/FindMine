# Verifier proxy precision/recall (dinov2)

PROXY: synthetic views of COCO val50 crops, not real items. Replace with FindMine-Home numbers.

187 items, 93 test positives, 3623 test negatives (3158 same-class look-alikes).
Fusion thetas (bias, emb, geom, colour): [-3.88, 2.86, 3.5, 1.76]

| score | threshold | precision | recall | precision vs look-alikes only |
|---|---|---|---|---|
| embedding | 0.93 | 1.00 | 0.51 | 1.00 |
| geometry | 1.00 | 1.00 | 0.82 | 1.00 |
| colour | 0.95 | 0.24 | 0.51 | 0.25 |
| fused | 0.50 | 1.00 | 0.99 | 1.00 |
| fused | 0.80 | 1.00 | 0.96 | 1.00 |

Single cues are thresholded at their median test-positive score (recall about 0.5) for comparison.

## Caveat seen in `server/test_verify.py`

A plain dark TV screen, re-photographed at a slight angle and 30% darker, scores only 0.15 (look-alike TV: 0.08).
Cues: embedding 0.64 vs 0.35, geometry 0 (about 20 SIFT points on a flat screen), colour 0.17 (darkening shifts the HSV histogram).
The proxy views are easier than real photos, so these weights expect high embedding and geometry scores. Ranking still works,
but absolute confidences on plain, low-texture items will be too low. **Refit on FindMine-Home photos before setting the found threshold.**
