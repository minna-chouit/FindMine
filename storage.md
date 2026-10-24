# FindMine storage

## What we keep, and why

| Data | Kept? | Where | Why |
|---|---|---|---|
| **Registration photos** (5 per item) | ✅ Yes | S3 | The search compares what the camera sees with these photos. Without them, "is it mine?" can't work. |
| **Fingerprints** of those photos | ✅ Yes | S3 (`.npz`) | Computed once at registration, so the search doesn't recompute them every time. |
| **Item record** (name, type, consent, last seen) | ✅ Yes | DynamoDB | One row per item. |
| **Live search frames** | ❌ No | — | Only processed in memory. Privacy rule: never stored by default. |
| Search traces (decisions, scores) | In memory + server logs | `/trace` | Debugging and the report. No pictures. |

## S3 layout: `s3://findmine-items-bucket/`

```
items/<item_id>/ref0-front.jpg    photo         metadata: item-id, class, side, train-ok
items/<item_id>/ref0-front.npz    fingerprint   same metadata
items/<item_id>/ref1-back.jpg
items/<item_id>/ref1-back.npz
items/<item_id>/ref2-left.jpg  …  ref3-right  …  ref4-top
models/findmine12.onnx            detector (used by infra/deploy.sh)
models/dinov2_vits14.onnx         verifier
```

Each file name says **which photo** it is (`ref2-left`), and each file carries **S3 metadata** (`class=keys`, `side=left`, `train-ok=true`), so the bucket makes sense even without the database. Items registered before this change are named `ref0.jpg` … and have no metadata; they still work.

## DynamoDB table: `findmine-items`

| Field | Example | Meaning |
|---|---|---|
| `item_id` (key) | `d55dce134dd0` | |
| `name` | `my keys` | What the user calls it |
| `class` | `keys` | One of the 12 detector classes |
| `n_refs` | `5` | Photos saved |
| `refs` | `[{jpg, npz, points, side}]` | One entry per photo |
| `train_ok` | `true` | User allowed these photos to improve the models |
| `created_at` | `1792613887` | Unix time |
| `last_seen` | `14:02 on 23 Oct` | Set by **Got it** |

## Using user photos to improve the models

Only with consent. The register page has an **unticked** box: *"Allow FindMine to use these photos to improve the app (optional)"*.

```
.venv/bin/python ml/export_training.py      # → data/user_refs/<class>/… + manifest.csv (consented items only)
```

- **Verifier** (is it mine?): registration photos are exactly the right data to refit the fusion weights and the ask/found thresholds.
- **Detector** (is there a key?): the photos are close-ups, so they need boxes drawn before training. Home videos/photos from the team are still the main source for small objects.
- Nothing retrains automatically. Training is a manual step (Kaggle/NVIDIA → new `.onnx` → replace on the server).

## Still to do

- **Delete my data** (Day 10): one button that removes an item's S3 folder and its DynamoDB row.
- **Face blur** (Day 10): blur faces in registration photos before they are saved (project rule for anything stored).
- Optional: S3 lifecycle rule to delete items not used for a year.
