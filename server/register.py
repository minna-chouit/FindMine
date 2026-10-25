"""Item registration: guided photos -> S3 (JPEG + features) + DynamoDB record.

Endpoints
  POST /items                      form: name, cls (optional)  -> new item
  POST /items/{item_id}/photos     form: photo (JPEG)          -> saves one reference photo
  GET  /items                                                  -> list of items
  GET  /items/{item_id}                                        -> one item
  DELETE /items/{item_id}?owner=…                              -> deletes the item, its photos and fingerprints
  DELETE /items?owner=…                                        -> "delete my data": every item of this phone

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
from pipeline.models import detector, face_blur, names, verifier

REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET = os.environ.get("FINDMINE_BUCKET", "findmine-items-bucket")
TABLE = os.environ.get("FINDMINE_TABLE", "findmine-items")
MAX_REFS = 5
SIDES = ("front", "back", "left", "right", "top")  # the 5 guided registration photos
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
def create_item(name: str = Form(...), cls: str = Form(""), owner: str = Form(""), train_ok: str = Form("false")):
    name = name.strip()
    if not name:
        raise HTTPException(400, "name is empty")
    item = {
        "item_id": uuid.uuid4().hex[:12],
        "name": name,
        "class": cls.strip(),
        "owner": owner.strip(),  # random id of the phone that registered it
        # Consent: may these photos be used to improve the models? Off unless the user ticks the box.
        "train_ok": train_ok.strip().lower() in ("true", "1", "yes", "on"),
        "n_refs": 0,
        "refs": [],
        "created_at": int(time.time()),
    }
    table.put_item(Item=item)
    return _public(item)


# async on purpose: requests run one at a time on the event loop, so the shared
# Verifier network is never used by two requests at once.
@router.post("/items/{item_id}/photos")
async def add_photo(item_id: str, photo: UploadFile = File(...), side: str = Form("")):
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
    img, n_faces = face_blur(img)  # plan F8.2: no face is ever stored (a helper or family in the background)
    if n_faces:
        data = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    feats = verifier.features(item_crop(img, item.get("class", "")))
    pts = feats.get("pts")
    n_pts = 0 if pts is None else int(len(pts))
    if feats.get("des") is None or n_pts < MIN_POINTS:
        return {"ok": False, "retake": True, "points": n_pts,
                "message": "Not enough detail. Move closer or try another angle."}

    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: v for k, v in feats.items() if v is not None})
    side = side if side in SIDES else "other"
    jpg_key = f"items/{item_id}/ref{n}-{side}.jpg"
    npz_key = f"items/{item_id}/ref{n}-{side}.npz"
    # Labels travel with every file, so S3 can be browsed or exported without the database.
    meta = {"item-id": item_id, "class": item.get("class") or "unknown", "side": side,
            "train-ok": "true" if item.get("train_ok") else "false"}
    s3.put_object(Bucket=BUCKET, Key=jpg_key, Body=data, ContentType="image/jpeg", Metadata=meta)
    s3.put_object(Bucket=BUCKET, Key=npz_key, Body=buf.getvalue(), Metadata=meta)

    table.update_item(
        Key={"item_id": item_id},
        UpdateExpression="SET refs = list_append(refs, :r), n_refs = n_refs + :one",
        ConditionExpression="n_refs = :n",
        ExpressionAttributeValues={
            ":r": [{"jpg": jpg_key, "npz": npz_key, "points": n_pts, "side": side}],
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


def _delete_item(item):
    """Remove an item everywhere: every S3 object under items/<id>/ (photos + fingerprints), the DynamoDB row,
    and the in-memory fingerprint cache. Returns how many S3 objects were removed."""
    item_id, removed = item["item_id"], 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"items/{item_id}/"):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
            removed += len(keys)
    table.delete_item(Key={"item_id": item_id})
    for k in [k for k in _refs_cache if k[0] == item_id]:
        del _refs_cache[k]
    return removed


@router.delete("/items/{item_id}")
def delete_item(item_id: str, owner: str = ""):
    """Delete one item. A phone can only delete items it registered (items with no owner: anyone, old data)."""
    item = table.get_item(Key={"item_id": item_id}).get("Item")
    if not item:
        raise HTTPException(404, "item not found")
    if item.get("owner") and item["owner"] != owner:
        raise HTTPException(403, "this item belongs to another phone")
    return {"deleted": [item_id], "files_removed": _delete_item(item)}


@router.delete("/items")
def delete_my_items(owner: str = ""):
    """Delete my data: every item registered from this phone. Needs the owner id, so nobody can wipe everything."""
    owner = owner.strip()
    if not owner:
        raise HTTPException(400, "owner is required")
    mine = [i for i in table.scan().get("Items", []) if i.get("owner") == owner]
    files = sum(_delete_item(i) for i in mine)
    return {"deleted": [i["item_id"] for i in mine], "files_removed": files}


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
