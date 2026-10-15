import numpy as np
from openarm_safety_bridge.visibility_memory import VisibilityMemory
from openarm_safety_bridge.visibility_memory import memory_collision_boxes


def test_visible_occupied_surface_refreshes_age_without_semantic_detection():
    memory = occupied()
    memory.carve(np.array([[0., 0., 1.01]]), np.eye(3), np.zeros(3), 90.)
    _, ages, uncertainty = memory.snapshot(90.)
    assert np.all(ages == 0) and np.all(uncertainty == 0)


def test_memory_boxes_preserve_gap_and_localize_old_uncertainty():
    boxes = memory_collision_boxes(np.array([[0, 0, 0], [1, 0, 0]]), np.array([.12, 0.]))
    assert len(boxes) == 2
    assert boxes[0][1][0] < .3
    assert boxes[1][1][0] < .05


def occupied():
    memory = VisibilityMemory()
    memory.observe(np.array([[0., 0., 1.]]), 1., np.zeros(3))
    return memory


def test_empty_and_occluded_rays_never_clear_even_after_expiry():
    memory = occupied()
    for stamp in range(2, 10):
        memory.carve(np.array([[0., 0., .4]]), np.eye(3), np.zeros(3), stamp)
        memory.observe(np.empty((0, 3)), stamp, np.zeros(3))
    points, ages, uncertainty = memory.snapshot(100.)
    assert len(points) == 2 and min(ages) == 99.
    assert min(uncertainty) == .005


def test_camera_turn_away_retains_world_position():
    memory = occupied()
    rotation = np.diag([-1., 1., -1.])
    for stamp in range(2, 6):
        memory.carve(np.array([[0., 0., 2.]]), rotation, np.zeros(3), stamp)
    np.testing.assert_allclose(memory.snapshot(6.)[0], [[0, 0, 1], [0, 0, 1]])


def test_only_repeated_new_visible_free_rays_can_clear():
    memory = occupied()
    far = np.array([[0., 0., 2.]])
    for _ in range(5):
        memory.carve(far, np.eye(3), np.zeros(3), 2.)
    assert memory.cells
    memory.carve(far, np.eye(3), np.zeros(3), .5)
    assert memory.cells
    for stamp in [3., 4.]:
        memory.carve(far, np.eye(3), np.zeros(3), stamp)
    assert not memory.cells


def test_nearest_return_protects_against_clearing_through_robot():
    memory = occupied()
    for stamp in range(2, 6):
        memory.carve(np.array([[0, 0, .5], [0, 0, 2.]]), np.eye(3), np.zeros(3), stamp)
    assert memory.cells


def test_capacity_overflow_remains_explicit_unknown_and_prediction_bounded():
    memory = VisibilityMemory(capacity=4)
    memory.observe(np.arange(30).reshape(-1, 3) / 10, 1., np.array([.2, 0, 0]))
    assert len(memory.cells) == 4 and memory.capacity_exceeded
    points, _, _ = memory.snapshot(1000.)
    np.testing.assert_allclose(points[4:] - points[:4], np.tile([.1, 0, 0], (4, 1)))
