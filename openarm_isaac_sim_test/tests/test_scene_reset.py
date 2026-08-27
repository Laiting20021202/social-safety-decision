from openarm_sim.config import load_yaml
from openarm_sim.scene_model import deterministic_cube_layout


def test_scene_seed_reproduces_identical_cube_layout() -> None:
    config = load_yaml("config/scene.yaml")
    first = deterministic_cube_layout(config)
    second = deterministic_cube_layout(config)
    assert first == second
    assert len(first) == 2
    assert {cube.color for cube in first} == {"left", "right"}
    assert {cube.name for cube in first} == {"left_target_cube", "right_target_cube"}


def test_cube_spacing_constraint() -> None:
    config = load_yaml("config/scene.yaml")
    cubes = deterministic_cube_layout(config)
    separation = float(config["target_cubes"]["size"])
    for index, cube in enumerate(cubes):
        for other in cubes[index + 1 :]:
            dx = cube.position[0] - other.position[0]
            dy = cube.position[1] - other.position[1]
            assert (dx * dx + dy * dy) ** 0.5 >= separation
