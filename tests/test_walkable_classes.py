from __future__ import annotations

import numpy as np

from social_bev.segmentation import merge_walkable_classes


def test_walkable_class_merge_respects_blocked_classes() -> None:
    label_map = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)
    id2label = {0: "floor", 1: "wall", 2: "road"}
    class_config = {
        "walkable_classes": ["floor", "road", "wall"],
        "optional_walkable_classes": [],
        "blocked_classes": ["wall"],
    }
    mask, labels = merge_walkable_classes(label_map, id2label, class_config)
    assert mask.tolist() == [[True, False, True], [True, False, True]]
    assert set(labels) == {"floor", "road", "wall"}

