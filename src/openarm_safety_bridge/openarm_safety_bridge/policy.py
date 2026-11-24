from __future__ import annotations

from dataclasses import dataclass

from openarm_sim.state_machine import SafetyState


@dataclass
class SafetyPolicy:
    warning_m: float
    pause_m: float
    emergency_m: float
    resume_m: float
    clear_duration_sec: float
    timeout_sec: float
    replan_delay_sec: float = 0.25
    emergency_confirmation_sec: float = 0.0
    state: SafetyState = SafetyState.SAFE
    last_observation_sec: float | None = None
    clear_since_sec: float | None = None
    transition_sec: float = 0.0
    estop_latched: bool = False
    emergency_since_sec: float | None = None
    escape_grace_until_sec: float = 0.0

    def observe(self, distance_m: float, now_sec: float) -> SafetyState:
        self.last_observation_sec = now_sec
        if distance_m <= self.emergency_m:
            if now_sec < self.escape_grace_until_sec and not self.estop_latched:
                # A dedicated upward/away trajectory has already passed
                # MoveIt's collision check.  Let that bounded maneuver run
                # instead of latching the arm at the pose being approached by
                # the hand.  Failure to make progress still reaches the hard
                # stop as soon as this finite grace interval expires.
                self.emergency_since_sec = None
                return self._transition(SafetyState.PAUSE, now_sec)
            if self.emergency_since_sec is None:
                self.emergency_since_sec = now_sec
            if (
                now_sec - self.emergency_since_sec
                >= max(self.emergency_confirmation_sec, 0.0)
            ):
                self.estop_latched = True
                return self._transition(SafetyState.EMERGENCY_STOP, now_sec)
            # Stop immediately, but require a short run of confirmed near
            # points before permanently latching a perception E-stop.
            return self._transition(SafetyState.PAUSE, now_sec)
        self.emergency_since_sec = None
        if self.estop_latched:
            return self.state
        if distance_m <= self.pause_m:
            self.clear_since_sec = None
            if self.state is SafetyState.REPLAN:
                return self.state
            if (
                self.state is SafetyState.PAUSE
                and now_sec - self.transition_sec >= self.replan_delay_sec
            ):
                return self._transition(SafetyState.REPLAN, now_sec)
            return self._transition(SafetyState.PAUSE, now_sec)
        if distance_m <= self.warning_m:
            self.clear_since_sec = None
            return self._transition(SafetyState.WARNING, now_sec)
        if distance_m < self.resume_m:
            self.clear_since_sec = None
            return self.state
        if self.clear_since_sec is None:
            self.clear_since_sec = now_sec
        if now_sec - self.clear_since_sec < self.clear_duration_sec:
            return self.state
        if self.state is SafetyState.PAUSE:
            return self._transition(SafetyState.REPLAN, now_sec)
        if self.state is SafetyState.REPLAN and now_sec - self.transition_sec >= 0.1:
            return self._transition(SafetyState.RECOVER, now_sec)
        if self.state in {SafetyState.RECOVER, SafetyState.WARNING}:
            return self._transition(SafetyState.SAFE, now_sec)
        return self.state

    def check_timeout(self, now_sec: float) -> SafetyState:
        if self.last_observation_sec is None:
            return self.state
        if now_sec - self.last_observation_sec > self.timeout_sec:
            self.estop_latched = True
            return self._transition(SafetyState.EMERGENCY_STOP, now_sec)
        return self.state

    def reset_estop(self, now_sec: float) -> SafetyState:
        self.estop_latched = False
        self.emergency_since_sec = None
        self.escape_grace_until_sec = 0.0
        self.clear_since_sec = now_sec
        return self._transition(SafetyState.REPLAN, now_sec)

    def grant_escape_grace(self, now_sec: float, duration_sec: float) -> SafetyState:
        """Boundedly defer an E-stop while a checked escape is executing."""

        if self.estop_latched:
            return self.state
        self.escape_grace_until_sec = max(
            self.escape_grace_until_sec,
            float(now_sec) + max(float(duration_sec), 0.0),
        )
        self.emergency_since_sec = None
        # Do not move REPLAN back to PAUSE: pose_goal may already be handing
        # the checked escape to the controller, and a fresh PAUSE transition
        # would immediately cancel that very trajectory.
        return self.state

    def _transition(self, state: SafetyState, now_sec: float) -> SafetyState:
        if state is not self.state:
            self.state = state
            self.transition_sec = now_sec
        return self.state
