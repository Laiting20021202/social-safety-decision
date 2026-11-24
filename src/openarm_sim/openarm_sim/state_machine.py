from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskState(str, Enum):
    HOME = "HOME"
    SELECT_OBJECT = "SELECT_OBJECT"
    PRE_GRASP = "PRE_GRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    TRANSIT = "TRANSIT"
    PLACE = "PLACE"
    RETREAT = "RETREAT"
    NEXT_OBJECT = "NEXT_OBJECT"
    DONE = "DONE"


class SafetyState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    PAUSE = "PAUSE"
    REPLAN = "REPLAN"
    RECOVER = "RECOVER"
    EMERGENCY_STOP = "EMERGENCY_STOP"


TASK_SEQUENCE = tuple(TaskState)


@dataclass
class TaskStateMachine:
    task_state: TaskState = TaskState.HOME
    safety_state: SafetyState = SafetyState.SAFE
    paused_from: TaskState | None = None

    def advance(self) -> TaskState:
        if self.safety_state in {SafetyState.PAUSE, SafetyState.EMERGENCY_STOP}:
            raise RuntimeError(f"cannot advance while safety state is {self.safety_state.value}")
        index = TASK_SEQUENCE.index(self.task_state)
        if self.task_state is not TaskState.DONE:
            self.task_state = TASK_SEQUENCE[index + 1]
        return self.task_state

    def set_safety(self, state: SafetyState | str) -> SafetyState:
        next_state = SafetyState(state)
        if next_state in {SafetyState.PAUSE, SafetyState.EMERGENCY_STOP}:
            self.paused_from = self.task_state
        if self.safety_state is SafetyState.EMERGENCY_STOP and next_state is SafetyState.SAFE:
            raise RuntimeError("EMERGENCY_STOP requires an explicit RECOVER transition")
        self.safety_state = next_state
        return self.safety_state

    def reset_for_next_object(self) -> TaskState:
        if self.task_state is not TaskState.NEXT_OBJECT:
            raise RuntimeError("next-object reset is only valid from NEXT_OBJECT")
        self.task_state = TaskState.SELECT_OBJECT
        return self.task_state

