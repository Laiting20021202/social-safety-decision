from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import numpy as np

from realtime_safety.types import PerformanceSnapshot, SafetySnapshot


class SessionLogger:
    """Flush-on-update JSONL/CSV logger; it retains no in-memory session history."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.output_dir / "safety.jsonl").open("a", encoding="utf-8", buffering=1)
        self._trajectory = (self.output_dir / "trajectories.csv").open("a", encoding="utf-8", newline="", buffering=1)
        self._csv = csv.writer(self._trajectory)
        if self._trajectory.tell() == 0:
            self._csv.writerow(
                ["timestamp", "frame_index", "track_id", "class_name", "scale_unit", "x", "y", "z", "vx", "vy", "vz", "risk_score", "ttc"]
            )
        self._lock = threading.Lock()
        self._closed = False

    def log(
        self,
        snapshot: SafetySnapshot,
        profile: str,
        depth_mode: str,
        scale_mode: str,
        performance: PerformanceSnapshot,
    ) -> None:
        if self._closed:
            return
        zones = {zone.track_id: zone for zone in snapshot.danger_zones}
        objects = []
        scale_unit = "metric" if snapshot.metric_valid else "relative"
        for track in snapshot.tracks:
            zone = zones.get(track.track_id)
            item = {
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": track.confidence,
                "position_xyz": _array(track.position_xyz),
                "velocity_xyz": _array(track.velocity_xyz),
                "velocity_unit": "m/s" if snapshot.metric_valid else "relative_units/s",
                "bbox3d": {"minimum": _array(track.bbox3d.minimum), "maximum": _array(track.bbox3d.maximum)},
                "motion_state": track.motion_state,
                "risk_score": zone.risk_score if zone else 0.0,
                "ttc": zone.ttc if zone else None,
                "predicted_positions": _array(zone.predicted_positions) if zone else [],
            }
            objects.append(item)
        record = {
            "timestamp": snapshot.timestamp,
            "frame_index": snapshot.frame_index,
            "profile": profile,
            "depth_mode": depth_mode,
            "scale_mode": scale_mode,
            "metric_valid": snapshot.metric_valid,
            "safety_state": snapshot.safety_state.value,
            "recommended_action": snapshot.recommended_action.value,
            "degraded_reasons": snapshot.degraded_reasons,
            "objects": objects,
            "performance": {
                "input_fps": performance.input_fps,
                "display_fps": performance.display_fps,
                "segmentation_fps": performance.segmentation_fps,
                "reconstruction_fps": performance.reconstruction_fps,
                "safety_fps": performance.safety_fps,
                "latency_ms": performance.average_latency_ms,
                "p95_latency_ms": performance.p95_latency_ms,
                "dropped_frames": performance.dropped_frames,
                "queue_size": performance.queue_size,
                "ram_mb": performance.ram_mb,
                "vram_used_mb": performance.vram_used_mb,
            },
        }
        with self._lock:
            self._jsonl.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            for track in snapshot.tracks:
                zone = zones.get(track.track_id)
                self._csv.writerow(
                    [
                        snapshot.timestamp,
                        snapshot.frame_index,
                        track.track_id,
                        track.class_name,
                        scale_unit,
                        *map(float, track.position_xyz),
                        *map(float, track.velocity_xyz),
                        zone.risk_score if zone else 0.0,
                        zone.ttc if zone else "",
                    ]
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._jsonl.close()
            self._trajectory.close()

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _array(value: np.ndarray) -> list:
    return np.asarray(value).tolist()
