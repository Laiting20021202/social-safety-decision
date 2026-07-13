from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from realtime_safety.config import GuiConfig
from realtime_safety.types import PerformanceSnapshot, SafetyLevel

LOGGER = logging.getLogger(__name__)


class Dashboard:
    """Viser video/control dashboard. 3D scene rendering is added by Scene3D."""

    def __init__(self, config: GuiConfig, on_command: Callable[[str, object | None], None]) -> None:
        import viser

        self.server = viser.ViserServer(host=config.host, port=config.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible", dark_mode=True)
        self._on_command = on_command
        self._lock = threading.Lock()
        self._upload_dir = Path(tempfile.mkdtemp(prefix="realtime_safety_upload_"))
        self._build_controls()
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Select a source and press Start", (92, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        with self.server.gui.add_folder("Live Video"):
            self.video = self.server.gui.add_image(placeholder[..., ::-1], label="Annotated RGB", format="jpeg", jpeg_quality=82)

    def _build_controls(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Input & Playback", expand_by_default=True):
            self.file_path = gui.add_text("File / RTSP / webcam", initial_value="")
            self.upload = gui.add_upload_button("Upload Video", mime_type="video/*")
            self.start = gui.add_button("Start", color="green")
            self.pause = gui.add_button("Pause / Resume")
            self.stop = gui.add_button("Stop", color="red")
            self.restart = gui.add_button("Restart")
            self.loop = gui.add_checkbox("Loop", initial_value=False)
            self.speed = gui.add_slider("Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0)
            self.seek = gui.add_number("Seek (seconds)", initial_value=0.0, min=0.0)
        self.start.on_click(lambda _: self._on_command("start", self.file_path.value))
        self.pause.on_click(lambda _: self._on_command("pause_resume", None))
        self.stop.on_click(lambda _: self._on_command("stop", None))
        self.restart.on_click(lambda _: self._on_command("restart", None))
        self.loop.on_update(lambda _: self._on_command("loop", self.loop.value))
        self.speed.on_update(lambda _: self._on_command("speed", self.speed.value))
        self.seek.on_update(lambda _: self._on_command("seek", self.seek.value))

        @self.upload.on_upload
        def _(event) -> None:
            uploaded = event.target.value
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                LOGGER.warning("Rejected uploaded extension: %s", suffix)
                return
            destination = self._upload_dir / Path(uploaded.name).name
            destination.write_bytes(uploaded.content)
            self.file_path.value = str(destination)
            self._on_command("start", str(destination))

        with gui.add_folder("Visibility"):
            self.visibility = {
                label: gui.add_checkbox(label, initial_value=True)
                for label in (
                    "Point Cloud", "Tracking Points", "Segmentation", "Static Obstacles",
                    "Dynamic Obstacles", "Bounding Boxes", "Velocity Arrows", "Trajectories",
                    "Danger Zones", "Ground Plane", "Traversable Region", "Candidate Paths", "Selected Path",
                )
            }
            for label, handle in self.visibility.items():
                handle.on_update(lambda _, key=label: self._on_command("visibility", (key, self.visibility[key].value)))

        with gui.add_folder("Performance", expand_by_default=True):
            self.performance = gui.add_markdown("Waiting for frames…")
        with gui.add_folder("Status", expand_by_default=True):
            self.status = gui.add_markdown("**System:** IDLE")

    def update_video(self, annotated_bgr: np.ndarray) -> None:
        with self._lock:
            self.video.image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    def update_performance(self, perf: PerformanceSnapshot) -> None:
        self.performance.content = (
            f"| Input | Display | Segmentation | 3D | Safety |\n"
            f"|---:|---:|---:|---:|---:|\n"
            f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.segmentation_fps:.1f} FPS | "
            f"{perf.reconstruction_fps:.1f} FPS | {perf.safety_fps:.1f} FPS |\n\n"
            f"Latency avg / p95: **{perf.average_latency_ms:.1f} / {perf.p95_latency_ms:.1f} ms**  \n"
            f"Dropped: **{perf.dropped_frames}** · Queue: **{perf.queue_size}/{perf.queue_capacity}**  \n"
            f"RAM: **{perf.ram_mb:.0f} MB** · VRAM: **{perf.vram_used_mb:.0f} MB**"
        )

    def update_status(
        self,
        level: SafetyLevel,
        profile: str,
        depth_mode: str,
        scale_mode: str,
        cuda_status: str,
        action: str,
    ) -> None:
        self.status.content = (
            f"# {level.value}\n"
            f"Recommended action: **{action}**  \n"
            f"Profile: **{profile}** · Depth: **{depth_mode}**  \n"
            f"Scale: **{scale_mode.upper()} SCALE** · CUDA: **{cuda_status}**"
        )

    def close(self) -> None:
        self.server.stop()
        shutil.rmtree(self._upload_dir, ignore_errors=True)
