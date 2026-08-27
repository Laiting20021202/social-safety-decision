from __future__ import annotations

import numpy as np

from openarm_perception_adapter import node


def test_xyz_points_accepts_humble_structured_cloud(monkeypatch) -> None:
    rows = np.array(
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        dtype={
            "names": ("x", "y", "z"),
            "formats": ("<f4", "<f4", "<f4"),
            "offsets": (0, 4, 8),
            "itemsize": 16,
        },
    )
    monkeypatch.setattr(node.point_cloud2, "read_points", lambda *args, **kwargs: rows)

    points = node._xyz_points(object())

    np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])
    assert points.dtype == np.float32

