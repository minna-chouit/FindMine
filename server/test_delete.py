"""Delete my data: one item, or every item of a phone; never another phone's items. AWS faked in memory.

Run from the repo root:  .venv/Scripts/python server/test_delete.py
"""
import sys
import types

# The ONNX models are not needed here: stand-ins keep the test fast and runnable without models/.
models = types.ModuleType("pipeline.models")
models.detector = models.face_blur = models.verifier = None
models.names = ["keys", "wallet", "phone", "glasses", "sunglasses", "remote", "earbuds", "medicine", "cup", "bottle",
                "bag", "watch"]
sys.modules.setdefault("pipeline.models", models)

import register  # noqa: E402
from fastapi import HTTPException  # noqa: E402


class FakeTable:
    def __init__(self):
        self.items = {}

    def put_item(self, Item):
        self.items[Item["item_id"]] = Item

    def get_item(self, Key):
        item = self.items.get(Key["item_id"])
        return {"Item": item} if item else {}

    def delete_item(self, Key):
        self.items.pop(Key["item_id"], None)

    def scan(self):
        return {"Items": list(self.items.values())}


class FakeS3:
    def __init__(self):
        self.objects = set()

    def get_paginator(self, _name):
        s3 = self

        class Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(s3.objects) if k.startswith(Prefix)]}
        return Pages()

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.discard(o["Key"])


register.table, register.s3 = FakeTable(), FakeS3()


def add(name, owner):
    item = register.create_item(name=name, cls="keys", owner=owner, train_ok="false")
    for i, side in enumerate(register.SIDES):
        register.s3.objects |= {f"items/{item['item_id']}/ref{i}-{side}.jpg", f"items/{item['item_id']}/ref{i}-{side}.npz"}
    register._refs_cache[(item["item_id"], 5)] = ["fingerprints"]
    return item["item_id"]


a1, a2, b1 = add("Sara - keys", "phoneA"), add("Sara - wallet", "phoneA"), add("Amine - keys", "phoneB")

# Another phone can't delete Sara's item
try:
    register.delete_item(a1, owner="phoneB")
    raise AssertionError("phoneB deleted phoneA's item")
except HTTPException as e:
    assert e.status_code == 403

# Sara deletes one item: its 10 files, row and cache are gone, the rest stays
out = register.delete_item(a1, owner="phoneA")
assert out == {"deleted": [a1], "files_removed": 10}, out
assert a1 not in register.table.items and not any(a1 in k for k in register.s3.objects)
assert (a1, 5) not in register._refs_cache
assert a2 in register.table.items and b1 in register.table.items

# "Delete my data" removes everything of phoneA and nothing of phoneB
out = register.delete_my_items(owner="phoneA")
assert out["deleted"] == [a2] and out["files_removed"] == 10, out
assert list(register.table.items) == [b1] and len(register.s3.objects) == 10

# No owner: refused, so one request can never wipe the whole table
try:
    register.delete_my_items(owner="")
    raise AssertionError("deleted without an owner")
except HTTPException as e:
    assert e.status_code == 400
assert list(register.table.items) == [b1]
print("DELETE MY DATA TEST PASSED")
