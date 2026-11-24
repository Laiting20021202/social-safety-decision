from __future__ import annotations

import pytest

from openarm_sorting_task.grasp import grasp_decision


def test_auto_tries_physical_twice_before_magnetic_fallback() -> None:
    assert grasp_decision(
        "auto", 1, 2, stalled=False, reached_goal=True
    ) == "retry"
    assert grasp_decision(
        "auto", 2, 2, stalled=False, reached_goal=True
    ) == "magnetic"


def test_stalled_close_is_recorded_as_physical_contact() -> None:
    assert grasp_decision(
        "auto", 1, 2, stalled=True, reached_goal=False
    ) == "physical"


def test_physical_mode_never_uses_magnetic_fallback() -> None:
    assert grasp_decision(
        "physical", 2, 2, stalled=False, reached_goal=True
    ) == "fail"


def test_invalid_grasp_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported grasp mode"):
        grasp_decision("teleport", 1, 2, stalled=False, reached_goal=True)

