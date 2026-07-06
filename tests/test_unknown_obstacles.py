from __future__ import annotations

import numpy as np

from social_bev.unknown_obstacles import UnknownObstacleExtractor


def test_unknown_obstacle_tiny_contour_removed() -> None:
    mask = np.ones((120, 160), dtype=bool)
    mask[90:93, 78:81] = False
    extractor = UnknownObstacleExtractor(
        {
            "enabled": True,
            "ground_roi_start_ratio": 0.35,
            "corridor_width_ratio": 0.8,
            "minimum_area": 50,
            "maximum_area_ratio": 0.5,
            "morphology_kernel": 1,
        }
    )
    assert extractor.extract(mask, [], mask.shape) == []


def test_unknown_obstacle_extracted_in_corridor() -> None:
    mask = np.ones((120, 160), dtype=bool)
    mask[72:105, 66:96] = False
    extractor = UnknownObstacleExtractor(
        {
            "enabled": True,
            "ground_roi_start_ratio": 0.35,
            "corridor_width_ratio": 0.8,
            "minimum_area": 100,
            "maximum_area_ratio": 0.5,
            "morphology_kernel": 1,
        }
    )
    regions = extractor.extract(mask, [], mask.shape)
    assert len(regions) == 1
    assert regions[0].category == "unknown_obstacle"
    assert regions[0].note == "RGB estimate"

