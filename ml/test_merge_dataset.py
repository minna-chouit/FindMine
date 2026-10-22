"""ml/.venv/Scripts/python ml/test_merge_dataset.py"""
from merge_dataset import CLASSES, merge

K, CUP = CLASSES.index("keys"), CLASSES.index("cup")
coco = {"categories": [{"id": 47, "name": "cup"}, {"id": 1, "name": "person"}],
        "annotations": [
            {"image_id": 1, "category_id": 47, "bbox": [0, 0, 50, 100]},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},                  # not a target
            {"image_id": 1, "category_id": 47, "bbox": [0, 0, 5, 5], "iscrowd": 1}]}    # crowd
lvis = {"categories": [{"id": 605, "name": "key"}, {"id": 344, "name": "cup"}],
        "images": [
            {"id": 1, "width": 100, "height": 200, "coco_url": "http://x/train2017/000000000001.jpg"},
            {"id": 2, "width": 100, "height": 100, "coco_url": "http://x/val2017/000000000002.jpg",
             "not_exhaustive_category_ids": [605]},                                      # dropped
            {"id": 3, "width": 100, "height": 100, "coco_url": "http://x/val2017/000000000003.jpg"}],
        "annotations": [
            {"image_id": 1, "category_id": 605, "bbox": [10, 20, 30, 40]},
            {"image_id": 1, "category_id": 344, "bbox": [0, 0, 50, 100]},              # LVIS cup ignored
            {"image_id": 2, "category_id": 605, "bbox": [0, 0, 10, 10]}]}

out = merge([coco], [lvis])
assert list(out) == [("train2017", "000000000001")], out          # img 2 not-exhaustive, img 3 empty
lines = sorted(r[2] for r in out["train2017", "000000000001"])
assert lines == sorted([f"{CUP} 0.250000 0.250000 0.500000 0.500000",
                        f"{K} 0.250000 0.200000 0.300000 0.200000"]), lines
print("ok")
