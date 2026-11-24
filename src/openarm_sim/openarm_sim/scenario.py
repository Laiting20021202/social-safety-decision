from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .config import ConfigError, deep_merge, load_yaml, require_keys


class MotionPhase(str, Enum):
    WAITING = "WAITING"
    MOVING_IN = "MOVING_IN"
    HOLDING = "HOLDING"
    MOVING_OUT = "MOVING_OUT"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True)
class ScenarioSample:
    position: tuple[float, float, float]
    phase: MotionPhase
    cycle: int


def bounded_step(
    current: tuple[float, float, float] | np.ndarray,
    target: tuple[float, float, float] | np.ndarray,
    speed_mps: float,
    dt: float,
) -> tuple[float, float, float]:
    """Move toward a manual target without exceeding the configured speed."""

    start = np.asarray(current, dtype=float).reshape(3)
    goal = np.asarray(target, dtype=float).reshape(3)
    maximum_step = float(speed_mps) * float(dt)
    if maximum_step <= 0.0:
        raise ValueError("speed_mps and dt must be positive")
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance <= maximum_step:
        result = goal
    else:
        result = start + delta * (maximum_step / distance)
    return tuple(float(value) for value in result)


def load_scenario(name: str, path: str = "config/hand_scenarios.yaml") -> dict[str, Any]:
    root = load_yaml(path)
    scenarios = root.get("scenarios", {})
    if name not in scenarios:
        raise ConfigError(f"unknown hand scenario: {name}")
    scenario = deep_merge(root.get("defaults", {}), scenarios[name])
    require_keys(
        scenario,
        ("path_waypoints", "linear_speed", "hold_duration", "withdraw_speed", "repeat_count"),
        f"scenario {name}",
    )
    if not scenario["path_waypoints"]:
        raise ConfigError(f"scenario {name} must contain at least one waypoint")
    return scenario


class HandTrajectory:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.waypoints = np.asarray(scenario["path_waypoints"], dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ConfigError("hand path_waypoints must be an N x 3 list")
        self.start_time = float(scenario.get("start_time", 0.0))
        self.trigger_state = scenario.get("start_trigger_state")
        self.speed_in = float(scenario["linear_speed"])
        self.speed_out = float(scenario["withdraw_speed"])
        self.hold_duration = float(scenario["hold_duration"])
        self.repeat_count = int(scenario["repeat_count"])
        if min(self.speed_in, self.speed_out) <= 0.0:
            raise ConfigError("hand speeds must be positive")
        self._triggered_at: float | None = None
        self._withdraw_started_at: float | None = None
        self._withdraw_start_position: np.ndarray | None = None

    @property
    def parked_position(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.waypoints[0])

    def reset(self) -> None:
        self._triggered_at = None
        self._withdraw_started_at = None
        self._withdraw_start_position = None

    def trigger(self, sim_time: float) -> None:
        if self._triggered_at is None:
            self._triggered_at = sim_time

    def withdraw(self, sim_time: float, current_position: tuple[float, float, float]) -> None:
        self._withdraw_started_at = sim_time
        self._withdraw_start_position = np.asarray(current_position, dtype=float)

    def sample(self, sim_time: float, task_state: str | None = None) -> ScenarioSample:
        if self._withdraw_started_at is not None and self._withdraw_start_position is not None:
            delta = self.waypoints[0] - self._withdraw_start_position
            distance = float(np.linalg.norm(delta))
            duration = distance / self.speed_out
            elapsed = max(0.0, sim_time - self._withdraw_started_at)
            if duration <= 0.0 or elapsed >= duration:
                return ScenarioSample(self.parked_position, MotionPhase.COMPLETE, 1)
            position = self._withdraw_start_position + (elapsed / duration) * delta
            return ScenarioSample(
                tuple(float(value) for value in position), MotionPhase.MOVING_OUT, 1
            )
        if self.repeat_count == 0:
            return ScenarioSample(self.parked_position, MotionPhase.COMPLETE, 0)
        if self._triggered_at is None:
            state_triggered = self.trigger_state is not None and task_state == self.trigger_state
            time_triggered = self.trigger_state is None and sim_time >= self.start_time
            if state_triggered or time_triggered:
                self._triggered_at = sim_time
            else:
                return ScenarioSample(self.parked_position, MotionPhase.WAITING, 0)

        elapsed = max(0.0, sim_time - self._triggered_at)
        inward_duration = self._path_duration(self.speed_in)
        outward_duration = self._path_duration(self.speed_out)
        cycle_duration = inward_duration + self.hold_duration + outward_duration
        total_duration = cycle_duration * self.repeat_count
        if elapsed >= total_duration:
            return ScenarioSample(self.parked_position, MotionPhase.COMPLETE, self.repeat_count)
        cycle = int(elapsed // cycle_duration)
        local_time = elapsed - cycle * cycle_duration
        if local_time < inward_duration:
            position = self._sample_path(local_time, self.speed_in, reverse=False)
            phase = MotionPhase.MOVING_IN
        elif local_time < inward_duration + self.hold_duration:
            position = self.waypoints[-1]
            phase = MotionPhase.HOLDING
        else:
            outward_time = local_time - inward_duration - self.hold_duration
            position = self._sample_path(outward_time, self.speed_out, reverse=True)
            phase = MotionPhase.MOVING_OUT
        return ScenarioSample(tuple(float(value) for value in position), phase, cycle + 1)

    def _path_duration(self, speed: float) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1).sum() / speed)

    def _sample_path(self, elapsed: float, speed: float, reverse: bool) -> np.ndarray:
        points = self.waypoints[::-1] if reverse else self.waypoints
        remaining = elapsed * speed
        for start, end in zip(points[:-1], points[1:], strict=False):
            length = float(np.linalg.norm(end - start))
            if remaining <= length:
                ratio = 0.0 if length == 0.0 else remaining / length
                return start + ratio * (end - start)
            remaining -= length
        return points[-1]
