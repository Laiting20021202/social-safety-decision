from __future__ import annotations


def grasp_decision(
    mode: str,
    attempt: int,
    maximum_attempts: int,
    *,
    stalled: bool,
    reached_goal: bool,
) -> str:
    """Return physical, retry, magnetic, or fail for one close result."""

    if mode not in {"physical", "magnetic", "auto"}:
        raise ValueError(f"unsupported grasp mode: {mode}")
    if attempt < 1 or maximum_attempts < 1:
        raise ValueError("grasp attempts must be positive")
    if mode == "magnetic":
        return "magnetic"
    if stalled and not reached_goal:
        return "physical"
    if attempt < maximum_attempts:
        return "retry"
    return "magnetic" if mode == "auto" else "fail"

