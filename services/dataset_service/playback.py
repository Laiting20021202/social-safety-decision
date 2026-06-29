from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from packages.common_models import FramePacket, PlaybackState, ScenarioInfo
from packages.frame_sources import FrameSource


class PlaybackConfigUpdate(BaseModel):
    scenario_id: str | None = None
    speed: float | None = Field(default=None, gt=0)
    loop: bool | None = None
    step_mode: bool | None = None
    realtime_mode: bool | None = None
    experiment_mode: bool | None = None


class SeekRequest(BaseModel):
    scenario_id: str | None = None
    frame_index: int = Field(ge=0)


class StartPlaybackRequest(PlaybackConfigUpdate):
    frame_index: int | None = Field(default=None, ge=0)


@dataclass
class _Clock:
    wall_time: float
    frame_timestamp: float


class PlaybackManager:
    def __init__(self, frame_source: FrameSource) -> None:
        self.frame_source = frame_source
        self.state = PlaybackState()
        self._clock = _Clock(wall_time=time.monotonic(), frame_timestamp=0.0)

    def configure(self, update: PlaybackConfigUpdate) -> PlaybackState:
        if update.scenario_id is not None and update.scenario_id != self.state.scenario_id:
            self.load_scenario(update.scenario_id)
        if update.speed is not None:
            self.state.speed = update.speed
        if update.loop is not None:
            self.state.loop = update.loop
        if update.step_mode is not None:
            self.state.step_mode = update.step_mode
        if update.realtime_mode is not None:
            self.state.realtime_mode = update.realtime_mode
        if update.experiment_mode is not None:
            self.state.experiment_mode = update.experiment_mode
        self._touch()
        return self.state

    def load_scenario(self, scenario_id: str) -> PlaybackState:
        scenario = self.frame_source.get_scenario(scenario_id)
        self.state = PlaybackState(
            scenario_id=scenario.scenario_id,
            status="paused",
            frame_index=0,
            timestamp_sec=0.0,
            speed=self.state.speed,
            loop=self.state.loop,
            step_mode=self.state.step_mode,
            realtime_mode=self.state.realtime_mode,
            experiment_mode=self.state.experiment_mode,
            total_frames=scenario.frame_count,
            duration_sec=scenario.duration_sec,
        )
        self._reset_clock()
        return self.state

    def start(self, request: StartPlaybackRequest | None = None) -> PlaybackState:
        if request is not None:
            self.configure(request)
            if request.frame_index is not None:
                self.seek(
                    SeekRequest(
                        scenario_id=request.scenario_id,
                        frame_index=request.frame_index,
                    )
                )
        if self.state.scenario_id is None:
            scenarios = self.frame_source.list_scenarios()
            if not scenarios:
                raise FileNotFoundError("No scenarios available")
            self.load_scenario(scenarios[0].scenario_id)
        self.state.status = "playing"
        self._reset_clock()
        self._touch()
        return self.state

    def pause(self) -> PlaybackState:
        self.advance_to_now()
        self.state.status = "paused"
        self._touch()
        return self.state

    def stop(self) -> PlaybackState:
        self.state.status = "stopped"
        self.state.frame_index = 0
        self.state.timestamp_sec = 0.0
        self._reset_clock()
        self._touch()
        return self.state

    def reset(self) -> PlaybackState:
        scenario_id = self.state.scenario_id
        if scenario_id is None:
            self.state = PlaybackState()
        else:
            self.load_scenario(scenario_id)
            self.state.status = "paused"
        self._touch()
        return self.state

    def seek(self, request: SeekRequest) -> PlaybackState:
        if request.scenario_id is not None and request.scenario_id != self.state.scenario_id:
            self.load_scenario(request.scenario_id)
        scenario = self._require_scenario()
        frame_index = min(request.frame_index, max(0, scenario.frame_count - 1))
        frame = self.frame_source.get_frame(scenario.scenario_id, frame_index)
        self.state.frame_index = frame.frame_index
        self.state.timestamp_sec = frame.timestamp_sec
        if self.state.status == "idle":
            self.state.status = "paused"
        self._reset_clock()
        self._touch()
        return self.state

    def step(self, delta: int = 1) -> PlaybackState:
        scenario = self._require_scenario()
        next_index = self.state.frame_index + delta
        if next_index >= scenario.frame_count:
            next_index = 0 if self.state.loop else scenario.frame_count - 1
            if not self.state.loop:
                self.state.status = "ended"
        if next_index < 0:
            next_index = scenario.frame_count - 1 if self.state.loop else 0
        return self.seek(SeekRequest(scenario_id=scenario.scenario_id, frame_index=next_index))

    def current_frame(self) -> FramePacket:
        scenario = self._require_scenario()
        return self.frame_source.get_frame(scenario.scenario_id, self.state.frame_index)

    def advance_to_now(self) -> PlaybackState:
        if self.state.status != "playing" or self.state.step_mode:
            return self.state
        scenario = self._require_scenario()
        if scenario.frame_count <= 1:
            return self.state
        elapsed = (time.monotonic() - self._clock.wall_time) * self.state.speed
        target_timestamp = self._clock.frame_timestamp + elapsed
        frame_index = self._frame_index_at_timestamp(scenario, target_timestamp)
        if frame_index >= scenario.frame_count:
            if self.state.loop:
                frame_index = frame_index % scenario.frame_count
            else:
                frame_index = scenario.frame_count - 1
                self.state.status = "ended"
        frame = self.frame_source.get_frame(scenario.scenario_id, frame_index)
        self.state.frame_index = frame.frame_index
        self.state.timestamp_sec = frame.timestamp_sec
        self._touch()
        return self.state

    def snapshot(self) -> dict[str, object]:
        self.advance_to_now()
        frame = self.current_frame() if self.state.scenario_id else None
        return {
            "state": self.state.model_dump(mode="json"),
            "frame": frame.model_dump(mode="json") if frame else None,
        }

    def _require_scenario(self) -> ScenarioInfo:
        if self.state.scenario_id is None:
            scenarios = self.frame_source.list_scenarios()
            if not scenarios:
                raise FileNotFoundError("No scenarios available")
            self.load_scenario(scenarios[0].scenario_id)
        assert self.state.scenario_id is not None
        return self.frame_source.get_scenario(self.state.scenario_id)

    def _frame_index_at_timestamp(self, scenario: ScenarioInfo, timestamp_sec: float) -> int:
        interval = 0.5
        if scenario.frame_count > 1 and scenario.duration_sec > 0:
            interval = scenario.duration_sec / (scenario.frame_count - 1)
        return int(timestamp_sec / max(interval, 1e-6))

    def _reset_clock(self) -> None:
        self._clock = _Clock(wall_time=time.monotonic(), frame_timestamp=self.state.timestamp_sec)

    def _touch(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc)
