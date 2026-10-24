"""Item registration: guided photos -> S3 (JPEG + features) + DynamoDB record.

Endpoints
  POST /items                      form: name, cls (optional)  -> new item
  POST /items/{item_id}/photos     form: photo (JPEG)          -> saves one reference photo
  GET  /items                                                  -> list of items
  GET  /items/{item_id}                                        -> one item

load_refs(item_id) returns the stored feature dicts, ready for verifier(...) during search.
"""
import io
import os
import time
import uuid

import boto3
import cv2 as cv
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from pipeline.detect import crop_to_item
from pipeline.models import detector, names, verifier

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("FINDMINE_BUCKET", "findmine-items-bucket")
TABLE = os.environ.get("FINDMINE_TABLE", "findmine-items")
MAX_REFS = 5
MIN_POINTS = 20  # fewer SIFT points than this = photo has too little texture

s3 = boto3.client("s3", region_name=REGION)
table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
router = APIRouter()


def _public(item):
    """DynamoDB returns numbers as Decimal; convert for JSON."""
    return {
        "item_id": item["item_id"],
        "name": item["name"],
        "class": item.get("class", ""),
        "n_refs": int(item["n_refs"]),
        "created_at": int(item["created_at"]),
    }


@router.post("/items")
def create_item(name: str = Form(...), cls: str = Form(""), owner: str = Form("")):
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is empty")
    item = {
        "item_id": uuid.uuid4().hex[:12],
        "name": name,
        "class": cls.strip(),
        "owner": owner.strip(),  # random id of the phone that registered it
        "n_refs": 0,
        "refs": [],
        "created_at": int(time.time()),
    }
    table.put_item(Item=item)
    return _public(item)


# async on purpose: requests run one at a time on the event loop, so the shared
# Verifier network is never used by two requests at once.
@router.post("/items/{item_id}/photos")
async def add_photo(item_id: str, photo: UploadFile = File(...)):
    item = table.get_item(Key={"item_id": item_id}).get("Item")
    if not item:
        raise HTTPException(404, "item not found")
    n = int(item["n_refs"])
    if n >= MAX_REFS:
        raise HTTPException(400, "this item already has 5 photos")

    data = await photo.read()
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a JPEG image")

    feats = verifier.features(item_crop(img, item.get("class", "")))
    pts = feats.get("pts")
    n_pts = 0 if pts is None else int(len(pts))
    if feats.get("des") is None or n_pts < MIN_POINTS:
        return {"ok": False, "retake": True, "points": n_pts,
                "message": "Not enough detail. Move closer or try another angle."}

    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in feats.items() if v is not None})
    jpg_key = f"items/{item_id}/ref{n}.jpg"
    npz_key = f"items/{item_id}/ref{n}.npz"
    s3.put_object(Bucket=BUCKET, Key=jpg_key, Body=data, ContentType="image/jpeg")
    s3.put_object(Bucket=BUCKET, Key=npz_key, Body=buf.getvalue())

    table.update_item(
        Key={"item_id": item_id},
        UpdateExpression="SET refs = list_append(refs, :r), n_refs = n_refs + :one",
        ConditionExpression="n_refs = :n",
        ExpressionAttributeValues={
            ":r": [{"jpg": jpg_key, "npz": npz_key, "points": n_pts}],
            ":one": 1,
            ":n": n,
        },
    )
    return {"ok": True, "index": n, "n_refs": n + 1, "points": n_pts, "done": n + 1 >= MAX_REFS}


@router.get("/items")
def list_items(owner: str = ""):
    """Items registered from this phone (?owner=<its id>); all items when no owner is given."""
    items = [i for i in table.scan().get("Items", []) if not owner or i.get("owner") == owner]
    return sorted((_public(i) for i in items), key=lambda i: i["created_at"], reverse=True)


@router.get("/items/{item_id}")
def get_item(item_id: str):
    item = table.get_item(Key={"item_id": item_id}).get("Item")
    if not item:
        raise HTTPException(404, "item not found")
    return _public(item)


def item_crop(img, cls):
    """Reference photo cropped to the detected item, like the live crops it is compared with (and as
    ml/eval_home.py calibrated the thresholds). Whole photo if the class is unknown or nothing is found."""
    return crop_to_item(img, detector, names.index(cls)) if cls in names else img


_refs_cache = {}  # (item_id, n_refs) -> feature dicts


def load_refs(item_id):
    """Feature dicts of an item's reference photos, for verifier(features(crop), refs).
    Recomputed from the stored JPEGs (cropped to the item) so items registered before cropping existed
    match the calibrated thresholds too; cached per item."""
    item = table.get_item(Key={"item_id": item_id}).get("Item")
    if not item:
        return []
    key = (item_id, int(item["n_refs"]))
    if key not in _refs_cache:
        refs = []
        for r in item.get("refs", []):
            body = s3.get_object(Bucket=BUCKET, Key=r["jpg"])["Body"].read()
            img = cv.imdecode(np.frombuffer(body, np.uint8), cv.IMREAD_COLOR)
            if img is not None:
                refs.append(verifier.features(item_crop(img, item.get("class", ""))))
        _refs_cache[key] = refs
    return _refs_cache[key]
