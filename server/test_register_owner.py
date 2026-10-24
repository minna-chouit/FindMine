"""Each phone only lists the items it registered (owner id); no owner = all items. AWS faked in memory.

Run from the repo root:  .venv/Scripts/python server/test_register_owner.py
"""
import register


class FakeTable:
    def __init__(self):
        self.items = {"old": {"item_id": "old", "name": "Minna's bottle", "class": "bottle", "n_refs": 5, "created_at": 1}}

    def put_item(self, Item):
        self.items[Item["item_id"]] = Item

    def scan(self):
        return {"Items": list(self.items.values())}


register.table = FakeTable()
sara = register.create_item(name="Sara - red mug", cls="cup", owner="phoneA")
register.create_item(name="Amine - keys", cls="keys", owner="phoneB")
assert [i["name"] for i in register.list_items(owner="phoneA")] == ["Sara - red mug"]
assert [i["name"] for i in register.list_items(owner="phoneB")] == ["Amine - keys"]
assert register.list_items(owner="phoneC") == []                       # a new phone sees nothing, not Minna's items
assert len(register.list_items()) == 3                                  # no owner given: everything (old behaviour)
assert register.table.items[sara["item_id"]]["owner"] == "phoneA"
print("REGISTER OWNER TEST PASSED")
