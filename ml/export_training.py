"""Export registration photos that users allowed for training (train_ok = true).

    .venv/bin/python ml/export_training.py            # -> data/user_refs/
    .venv/bin/python ml/export_training.py --all      # include items without consent (team's own test items only!)

Writes one folder per class plus a manifest:
    data/user_refs/<class>/<item_id>_ref0-front.jpg
    data/user_refs/manifest.csv   item_id, name, class, side, points, file

Registration photos are close-ups of one item, so they are best for refitting the verifier
(fusion weights, ask/found thresholds). For detector fine-tuning they still need boxes drawn.
Live search frames are never stored, so they are never exported.
"""
import argparse
import csv
import os
from pathlib import Path

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("FINDMINE_BUCKET", "findmine-items-bucket")
TABLE = os.environ.get("FINDMINE_TABLE", "findmine-items")


def items(table):
    kwargs = {}
    while True:
        page = table.scan(**kwargs)
        yield from page.get("Items", [])
        if "LastEvaluatedKey" not in page:
            return
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/user_refs")
    ap.add_argument("--all", action="store_true", help="ignore consent (only for the team's own test items)")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=REGION)
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], 0
    for it in items(table):
        if not (args.all or it.get("train_ok")):
            skipped += 1
            continue
        cls = it.get("class") or "unknown"
        for i, ref in enumerate(it.get("refs", [])):
            side = ref.get("side", "")
            name = f"{it['item_id']}_ref{i}{'-' + side if side else ''}.jpg"
            dest = out / cls / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                s3.download_file(BUCKET, ref["jpg"], str(dest))
            rows.append({"item_id": it["item_id"], "name": it["name"], "class": cls, "side": side,
                         "points": int(ref.get("points", 0)), "file": str(dest.relative_to(out))})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "name", "class", "side", "points", "file"])
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len(rows)} photos to {out}/ ({skipped} items skipped: no consent)")


if __name__ == "__main__":
    main()
