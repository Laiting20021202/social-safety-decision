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
from realtime_safety.types import Detection2D, FramePacket, PerformanceSnapshot, SafetyLevel

LOGGER = logging.getLogger(__name__)


class Dashboard:
    """Viser video/control dashboard. 3D scene rendering is added by Scene3D."""

    def __init__(
        self,
        config: GuiConfig,
        on_command: Callable[[str, object | None], None],
        reconstruction_only: bool = False,
        people_overlay: bool = False,
    ) -> None:
        import viser

        self.server = viser.ViserServer(host=config.host, port=config.port)
        self.reconstruction_only = reconstruction_only
        self.people_overlay = people_overlay
        self.max_video_width = config.max_video_width
        self.server.gui.configure_theme(
            titlebar_content=None,
            control_layout="collapsible",
            control_width="small" if reconstruction_only else "medium",
            dark_mode=not reconstruction_only,
            show_logo=not reconstruction_only,
            show_share_button=False,
        )
        self._on_command = on_command
        self._lock = threading.Lock()
        self._upload_dir = Path(tempfile.mkdtemp(prefix="realtime_safety_upload_"))
        self._build_controls()
        background = 245 if reconstruction_only else 0
        foreground = (55, 55, 55) if reconstruction_only else (220, 220, 220)
        placeholder = np.full((360, 640, 3), background, dtype=np.uint8)
        message = "Upload a video to reconstruct it in 4D" if reconstruction_only else "Select a source and press Start"
        cv2.putText(placeholder, message, (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.72, foreground, 2)
        with self.server.gui.add_folder("Live Video"):
            label = "RGB input" if reconstruction_only else "Annotated RGB"
            self.video = self.server.gui.add_image(placeholder[..., ::-1], label=label, format="jpeg", jpeg_quality=72)
        if self.people_overlay:
            with self.server.gui.add_folder("YOLO person detections", expand_by_default=True):
                self.yolo_people_status = self.server.gui.add_markdown("Waiting for a reconstructed frame…")
                self.yolo_people_image = self.server.gui.add_image(
                    placeholder[..., ::-1], label="Aligned YOLO frame", format="jpeg", jpeg_quality=80
                )

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

        self.visibility = {}
        if not self.reconstruction_only:
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
            initial_status = "**4D reconstruction:** waiting for a video" if self.reconstruction_only else "**System:** IDLE"
            self.status = gui.add_markdown(initial_status)

    def update_video(self, annotated_bgr: np.ndarray) -> None:
        with self._lock:
            display = self._resize_for_gui(annotated_bgr)
            self.video.image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    def update_performance(self, perf: PerformanceSnapshot) -> None:
        if self.reconstruction_only:
            headers = "| Input | Display | YOLO person | 4D reconstruction |" if self.people_overlay else "| Input | Display | 4D reconstruction |"
            divider = "|---:|---:|---:|---:|" if self.people_overlay else "|---:|---:|---:|"
            values = (
                f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.segmentation_fps:.1f} FPS | "
                f"{perf.reconstruction_fps:.1f} FPS |"
                if self.people_overlay
                else f"| {perf.input_fps:.1f} FPS | {perf.display_fps:.1f} FPS | {perf.reconstruction_fps:.1f} FPS |"
            )
            self.performance.content = (
                f"{headers}\n{divider}\n{values}\n\n"
                f"Dropped: **{perf.dropped_frames}** · Queue: **{perf.queue_size}/{perf.queue_capacity}**  \n"
                f"RAM: **{perf.ram_mb:.0f} MB** · VRAM: **{perf.vram_used_mb:.0f} MB**"
            )
            return
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

    def update_reconstruction_status(
        self,
        depth_mode: str,
        cuda_status: str,
        model_ready: bool,
        errors: dict[str, str],
        yolo_count: int = 0,
        people_3d_count: int = 0,
        yolo_ready: bool = False,
        held_3d_count: int = 0,
    ) -> None:
        if model_ready:
            state = "### 4D reconstruction\nModel: **ready**"
        elif errors:
            detail = next(iter(errors.values()))
            state = f"### Model error\n{detail}"
        else:
            state = "### Loading model\nThe RGB video remains available while St4RTrack loads."
        people = (
            f"  \nYOLO people: **{yolo_count}** · tracked 3D boxes: **{people_3d_count}** "
            f"(short hold: **{held_3d_count}**)  \n"
            f"YOLO model: **{'ready' if yolo_ready else 'loading'}**"
            if self.people_overlay
            else ""
        )
        self.status.content = f"{state}  \nBackend: **{depth_mode}** · CUDA: **{cuda_status}**{people}"

    def update_people_detections(self, frame: FramePacket, detections: list[Detection2D]) -> None:
        if not self.people_overlay:
            return
        canvas = self._resize_for_gui(frame.bgr)
        scale_x = canvas.shape[1] / frame.original_width
        scale_y = canvas.shape[0] / frame.original_height
        overlay = canvas.copy()
        for detection in detections:
            if detection.mask is not None:
                mask = cv2.resize(
                    detection.mask.astype(np.uint8),
                    (canvas.shape[1], canvas.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                overlay[mask] = (
                    0.35 * overlay[mask] + 0.65 * np.array((70, 210, 80), dtype=np.float32)
                )
        canvas = cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0.0)
        for detection in detections:
            x1, y1, x2, y2 = np.rint(
                detection.bbox_xyxy * np.array((scale_x, scale_y, scale_x, scale_y), dtype=np.float32)
            ).astype(int)
            confirmed = detection.track_hits >= 2
            color = (55, 220, 75) if confirmed else (0, 190, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            text = f"person #{detection.track_id or 0} {detection.confidence:.2f}"
            cv2.putText(canvas, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        self.yolo_people_image.image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if detections:
            rows = "\n".join(
                f"| {detection.track_id or 0} | {detection.confidence:.2f} | "
                f"{'confirmed' if detection.track_hits >= 2 else 'pending'} |"
                for detection in sorted(detections, key=lambda item: item.confidence, reverse=True)
            )
            self.yolo_people_status.content = (
                f"Frame **{frame.frame_index}** · YOLO people: **{len(detections)}**  \n"
                f"| ID | confidence | 3D gate |\n|---:|---:|:---|\n{rows}"
            )
        else:
            self.yolo_people_status.content = f"Frame **{frame.frame_index}** · **no YOLO people detected**"

    def _resize_for_gui(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] <= self.max_video_width:
            return image.copy()
        scale = self.max_video_width / image.shape[1]
        return cv2.resize(
            image,
            (self.max_video_width, max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def close(self) -> None:
        self.server.stop()
        shutil.rmtree(self._upload_dir, ignore_errors=True)
