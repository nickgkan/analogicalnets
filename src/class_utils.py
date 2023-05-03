SCAN_OBJ_CLASSES = sorted([
    "bag",
    "bed",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "pillow",
    "shelf",
    "sink",
    "sofa",
    "table",
    "toilet"
])


PARTNET_CLASSES = sorted([
    "Bag",
    "Bed",
    "Bottle",
    "Bowl",
    "Chair",
    "Clock",
    "Dishwasher",
    "Display",
    "Door",
    "Earphone",
    "Faucet",
    "Hat",
    "Keyboard",
    "Knife",
    "Lamp",
    "Laptop",
    "Microwave",
    "Mug",
    "Refrigerator",
    "Scissors",
    "StorageFurniture",
    "Table",
    "TrashCan",
    "Vase"
])


SPLITS = {
    "multicat12": [
        "Chair",
        "Display",
        "StorageFurniture",
        "Bottle",
        "Clock",
        "Door",
        "Earphone",
        "Faucet",
        "Knife",
        "Lamp",
        "TrashCan",
        "Vase"
    ],
    "multicat20": [
        "Bag",
        "Bottle",
        "Bowl",
        "Chair",
        "Clock",
        "Display",
        "Door",
        "Earphone",
        "Faucet",
        "Hat",
        "Keyboard",
        "Knife",
        "Lamp",
        "Laptop",
        "Microwave",
        "Mug",
        "Scissors",
        "StorageFurniture",
        "TrashCan",
        "Vase"
    ],
    "multicatnovel4": [
        "Table",
        "Bed",
        "Dishwasher",
        "Refrigerator"
    ],
    "scanobjectnn11": [
        "bag",
        "bin",
        "box",
        "cabinet",
        "chair",
        "desk",
        "door",
        "pillow",
        "shelf",
        "sink",
        "sofa"
    ],
    "scanobjectnn4novel": [
        "bed",
        "display",
        "table",
        "toilet"
    ]
}


LEVELS = {
    'partnet': [1, 2, 3],
    'scanobjectnn': [1]
}


CLASSES = {
    'partnet': PARTNET_CLASSES,
    'scanobjectnn': SCAN_OBJ_CLASSES
}
