from __future__ import annotations

"""Conservative seed gate for suddenly entering hand candidates.

This gate is intentionally behavioural, not semantic.  It suppresses fixed
workspace residuals (table, mouse, fixtures) and admits a new track only after
measured motion.  Once admitted, the candidate remains an obstacle while it is
stationary or briefly occluded.  A trained RGB hand detector should be layered
on top before the candidate is labelled as a confirmed human hand.
"""

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class SuddenIntrusionGateConfig:
    enabled: bool = False
    minimum_seed_speed: float = 0.04
    minimum_motion_hits: int = 2
    maximum_seed_age_frames: int = 12

    def __post_init__(self) -> None:
        if self.minimum_seed_speed < 0.0:
            raise ValueError("minimum_seed_speed cannot be negative")
        if self.minimum_motion_hits < 1:
            raise ValueError("minimum_motion_hits must be positive")
        if self.maximum_seed_age_frames < 1:
            raise ValueError("maximum_seed_age_frames must be positive")


class SuddenIntrusionGate:
    """Keep only tracks that entered after calibration with coherent motion."""

    def __init__(self, config: SuddenIntrusionGateConfig | None = None) -> None:
        self.config = config or SuddenIntrusionGateConfig()
        self._motion_hits: dict[int, int] = {}
        self._accepted: set[int] = set()

    def reset(self) -> None:
        self._motion_hits.clear()
        self._accepted.clear()

    @property
    def accepted_track_ids(self) -> frozenset[int]:
        return frozenset(self._accepted)

    def filter_tracks(
        self,
        tracks: Iterable[Any],
        *,
        baseline_ready: bool,
    ) -> list[Any]:
        values = list(tracks)
        if not self.config.enabled:
            return values
        if not baseline_ready:
            self.reset()
            return []

        live_ids = {
            int(track.track_id)
            for track in values
            if _state_name(track) != "DELETED"
        }
        self._accepted.intersection_update(live_ids)
        self._motion_hits = {
            track_id: hits
            for track_id, hits in self._motion_hits.items()
            if track_id in live_ids
        }

        for track in values:
            track_id = int(track.track_id)
            state = _state_name(track)
            if state in {"DELETED", "LOST"} or track_id in self._accepted:
                continue
            age = max(int(getattr(track, "age_frames", 0)), 0)
            speed = float(
                getattr(
                    track,
                    "speed",
                    np.linalg.norm(
                        np.asarray(getattr(track, "velocity"), dtype=np.float32)
                    ),
                )
            )
            if (
                age <= self.config.maximum_seed_age_frames
                and np.isfinite(speed)
                and speed >= self.config.minimum_seed_speed
            ):
                self._motion_hits[track_id] = self._motion_hits.get(track_id, 0) + 1
            else:
                self._motion_hits[track_id] = 0
            if (
                state == "CONFIRMED"
                and self._motion_hits[track_id]
                >= self.config.minimum_motion_hits
            ):
                self._accepted.add(track_id)

        return [
            track
            for track in values
            if int(track.track_id) in self._accepted
            and _state_name(track) != "DELETED"
        ]


def _state_name(track: Any) -> str:
    state = getattr(track, "state", "")
    return str(getattr(state, "value", state)).upper()


__all__ = ["SuddenIntrusionGate", "SuddenIntrusionGateConfig"]
