import numpy as np

from realtime_safety.gui.metric_bev import (
    fit_metric_bev,
    rasterize_metric_bev,
)


def _tilted_workplane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 80, 120
    u = np.linspace(-0.55, 0.55, width, dtype=np.float32)
    v = np.linspace(0.45, 1.25, height, dtype=np.float32)
    x, y = np.meshgrid(u, v)
    z = -0.62 + 0.28 * y
    pointmap = np.stack((x, y, z), axis=-1).astype(np.float32)
    # A raised rectangular object must not change the fitted tabletop.
    pointmap[42:58, 50:72, 2] += 0.11
    points = pointmap.reshape(-1, 3)
    colors = np.full((len(points), 3), (95, 110, 120), dtype=np.uint8)
    return pointmap, points, colors


def test_metric_bev_fits_tilted_plane_and_rectifies_height() -> None:
    pointmap, points, _ = _tilted_workplane()

    calibration = fit_metric_bev(pointmap)
    projected = calibration.project(points)
    table = np.ones(pointmap.shape[:2], dtype=bool)
    table[42:58, 50:72] = False

    assert calibration.inlier_ratio > 0.85
    assert calibration.rms_error_m < 0.003
    assert np.percentile(np.abs(projected[table.ravel(), 2]), 95) < 0.003
    assert np.median(projected[~table.ravel(), 2]) > 0.09


def test_metric_bev_raster_marks_height_and_edge_masks_independently() -> None:
    pointmap, points, colors = _tilted_workplane()
    calibration = fit_metric_bev(pointmap)
    edge_points = points.reshape(pointmap.shape)[46:53, 55:65].reshape(-1, 3)

    image = rasterize_metric_bev(
        calibration,
        points,
        colors,
        obstacle_height_m=0.04,
        edge_points=edge_points,
        maximum_side_px=320,
    )

    assert image.ndim == 3 and image.shape[2] == 3
    assert max(image.shape[:2]) == 320
    # Elevated geometry is orange/red and exact EdgeTAM points are pink.
    assert np.count_nonzero((image[..., 0] > 180) & (image[..., 1] < 110)) > 20
    assert np.count_nonzero(
        (image[..., 0] == 255)
        & (image[..., 1] == 30)
        & (image[..., 2] == 105)
    ) > 5
