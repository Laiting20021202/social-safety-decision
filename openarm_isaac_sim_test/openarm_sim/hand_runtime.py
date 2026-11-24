from __future__ import annotations

import numpy as np

from .config import load_yaml
from .scenario import HandTrajectory, MotionPhase, ScenarioSample, bounded_step, load_scenario


class HandRuntime:
    """Advance the scripted or manually positioned obstacle independently of Isaac and ROS."""

    def __init__(self, scenario: str) -> None:
        config = load_yaml("config/hand_scenarios.yaml")["defaults"]
        self.workspace_min = np.asarray(config["manual_workspace_min"], dtype=float)
        self.workspace_max = np.asarray(config["manual_workspace_max"], dtype=float)
        self.parked = np.asarray(HandTrajectory(load_scenario("no_obstacle")).parked_position)
        self.scenario = scenario
        self.trajectory = HandTrajectory(load_scenario(scenario))
        self.scenario_time = 0.0
        self.paused = False
        self.manual = False
        self.manual_speed = 0.3
        self.sample = self.trajectory.sample(0.0, task_state="HOME")
        self.target = np.asarray(self.sample.position, dtype=float)

    def _select_scenario(self, scenario: str) -> None:
        self.scenario = scenario
        self.trajectory = HandTrajectory(load_scenario(scenario))
        self.scenario_time = 0.0
        self.manual = False
        self.target = np.asarray(self.trajectory.parked_position)

    def step(
        self,
        dt: float,
        task_state: str,
        commands: list[str],
        manual_target: tuple[float, float, float] | None,
    ) -> tuple[ScenarioSample, bool]:
        if manual_target is not None:
            self.target = np.clip(manual_target, self.workspace_min, self.workspace_max)
        reset_world = False
        for command in commands:
            action, _, argument = command.lower().partition(":")
            if action == "scenario" and argument:
                self._select_scenario(argument)
            elif action == "manual" and argument:
                self.manual = argument == "on"
            elif action == "speed" and argument:
                speed = float(argument)
                if speed not in {0.1, 0.3, 0.6}:
                    raise ValueError(f"unsupported manual hand speed: {speed}")
                self.manual_speed = speed
            elif action in {"start", "trigger", "trigger_hand"}:
                self.trajectory.trigger(self.scenario_time)
            elif action == "withdraw":
                if self.manual:
                    self.target = self.parked.copy()
                else:
                    self.trajectory.withdraw(self.scenario_time, self.sample.position)
            elif action == "reset_hand":
                self.manual = True
                self.target = self.parked.copy()
            elif action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "reset":
                self._select_scenario(self.scenario)
                reset_world = True

        if not self.paused:
            self.scenario_time += dt
        if self.manual:
            position = bounded_step(self.sample.position, self.target, self.manual_speed, dt)
            moving = not np.allclose(position, self.target, atol=1e-6)
            phase = MotionPhase.MOVING_IN if moving else MotionPhase.HOLDING
            self.sample = ScenarioSample(position, phase, 0)
        else:
            self.sample = self.trajectory.sample(self.scenario_time, task_state=task_state)
        return self.sample, reset_world
