from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from social_bev.config import load_config
from social_bev.frame_source import FrameSource
from social_bev.output_writer import OutputWriter
from social_bev.pipeline import SocialNavigationPipeline


class SocialBEVGui:
    """Tkinter GUI for processing real local videos or real image sequences."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RGB Social Navigation BEV")
        self.root.geometry("1180x820")
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.preview_image: ImageTk.PhotoImage | None = None

        self.input_var = tk.StringVar(value="data/4309719-uhd_3840_2160_24fps.mp4")
        self.config_var = tk.StringVar(value="configs/default.yaml")
        self.calibration_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value="outputs/gui_result.mp4")
        self.jsonl_var = tk.StringVar(value="outputs/gui_results.jsonl")
        self.occupancy_var = tk.StringVar(value="outputs/gui_occupancy")
        self.stride_var = tk.IntVar(value=1)
        self.max_frames_var = tk.IntVar(value=0)
        self.realtime_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Select a real video, image folder, or SCAND sample directory.")
        self.stats_var = tk.StringVar(value="")

        self._build_layout()
        self.root.after(50, self._poll_queue)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        controls = ttk.LabelFrame(outer, text="Input and output", padding=10)
        controls.pack(fill=tk.X)
        self._path_row(controls, "Input", self.input_var, 0, self._browse_input)
        self._path_row(controls, "Config", self.config_var, 1, self._browse_config)
        self._path_row(controls, "Calibration", self.calibration_var, 2, self._browse_calibration)
        self._path_row(controls, "Output video", self.output_var, 3, self._save_video)
        self._path_row(controls, "Output JSONL", self.jsonl_var, 4, self._save_jsonl)
        self._path_row(controls, "Occupancy dir", self.occupancy_var, 5, self._browse_output_dir)

        runtime = ttk.Frame(controls)
        runtime.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(runtime, text="Stride").pack(side=tk.LEFT)
        ttk.Spinbox(runtime, from_=1, to=30, textvariable=self.stride_var, width=6).pack(side=tk.LEFT, padx=(4, 18))
        ttk.Label(runtime, text="Max frames (0 = all)").pack(side=tk.LEFT)
        ttk.Spinbox(runtime, from_=0, to=100000, textvariable=self.max_frames_var, width=8).pack(side=tk.LEFT, padx=(4, 18))
        ttk.Checkbutton(runtime, text="Realtime playback", variable=self.realtime_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Button(runtime, text="Start", command=self.start).pack(side=tk.LEFT, padx=4)
        self.pause_button = ttk.Button(runtime, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(runtime, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)

        status = ttk.LabelFrame(outer, text="Status", padding=8)
        status.pack(fill=tk.X, pady=(8, 8))
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")
        ttk.Label(status, textvariable=self.stats_var).pack(anchor="w")

        preview_frame = ttk.LabelFrame(outer, text="Live 2x2 result", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        row: int,
        command: Any,
    ) -> None:
        ttk.Label(parent, text=label, width=14).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=2, padx=(4, 4))
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", pady=2)
        parent.columnconfigure(1, weight=1)

    def _browse_input(self) -> None:
        choice = filedialog.askopenfilename(
            title="Select real video or image",
            filetypes=[
                ("Video/Image", "*.mp4 *.avi *.mov *.mkv *.webm *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not choice:
            directory = filedialog.askdirectory(title="Or select image directory / SCAND sample directory")
            choice = directory
        if choice:
            self.input_var.set(choice)

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(title="Select config YAML", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.config_var.set(path)

    def _browse_calibration(self) -> None:
        path = filedialog.askopenfilename(title="Select calibration YAML", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.calibration_var.set(path)

    def _save_video(self) -> None:
        path = filedialog.asksaveasfilename(title="Output video", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if path:
            self.output_var.set(path)

    def _save_jsonl(self) -> None:
        path = filedialog.asksaveasfilename(title="Output JSONL", defaultextension=".jsonl", filetypes=[("JSONL", "*.jsonl")])
        if path:
            self.jsonl_var.set(path)

    def _browse_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Output occupancy directory")
        if directory:
            self.occupancy_var.set(directory)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Already running", "Processing is already running.")
            return
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Select a real video, image, or image directory.")
            return
        if not input_path.isdigit() and not Path(input_path).exists():
            messagebox.showerror("Input not found", f"Input does not exist: {input_path}")
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.pause_button.configure(text="Pause", state=tk.NORMAL)
        self.status_var.set("Starting pipeline...")
        self.stats_var.set("")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.status_var.set("Stopping after current frame...")

    def toggle_pause(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Resuming...")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused. Recognition will stop after the current frame.")

    def _run_worker(self) -> None:
        writer: OutputWriter | None = None
        processed = 0
        try:
            config = load_config(self.config_var.get().strip() or "configs/default.yaml")
            config.setdefault("runtime", {})["display"] = False
            source = FrameSource(self.input_var.get().strip(), stride=int(self.stride_var.get()))
            pipeline = SocialNavigationPipeline(
                config=config,
                calibration_path=self.calibration_var.get().strip() or None,
            )
            max_frames = int(self.max_frames_var.get())
            start_time = time.perf_counter()
            playback_wall_start = time.perf_counter()
            playback_timestamp_start: float | None = None
            pause_offset = 0.0
            source_iter = iter(source)
            while not self.stop_event.is_set():
                pause_offset += self._wait_if_paused()
                if self.stop_event.is_set():
                    break
                try:
                    frame = next(source_iter)
                except StopIteration:
                    break
                pause_offset += self._wait_if_paused()
                if self.stop_event.is_set():
                    break
                if bool(self.realtime_var.get()):
                    if playback_timestamp_start is None:
                        playback_timestamp_start = frame.timestamp
                        playback_wall_start = time.perf_counter()
                    target_time = (
                        playback_wall_start
                        + pause_offset
                        + max(0.0, float(frame.timestamp) - playback_timestamp_start)
                    )
                    pause_offset += self._wait_until_realtime_target(target_time)
                    if self.stop_event.is_set():
                        break
                if writer is None:
                    writer = OutputWriter(
                        self.output_var.get().strip(),
                        self.jsonl_var.get().strip(),
                        self.occupancy_var.get().strip(),
                        fps=source.fps,
                    )
                result = pipeline.process_frame(frame.image, frame.timestamp)
                writer.write(result)
                processed += 1
                elapsed = max(1e-6, time.perf_counter() - start_time)
                self._put_latest(
                    {
                        "type": "frame",
                        "image": result.visualization,
                        "status": f"Processed {processed} real input frames",
                        "stats": (
                            f"FPS {result.fps:.2f}, AVG {processed / elapsed:.2f}, "
                            f"metric_bev={result.bev.metric_bev}, output={self.output_var.get()}"
                        ),
                    }
                )
                if max_frames > 0 and processed >= max_frames:
                    break
            self._put_latest({"type": "done", "status": f"Finished. Processed {processed} frames."})
        except Exception as exc:
            self._put_latest({"type": "error", "status": str(exc)})
        finally:
            if writer is not None:
                writer.close()

    def _wait_if_paused(self) -> float:
        if not self.pause_event.is_set():
            return 0.0
        paused_at = time.perf_counter()
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.05)
        return time.perf_counter() - paused_at

    def _wait_until_realtime_target(self, target_time: float) -> float:
        paused_total = 0.0
        while not self.stop_event.is_set():
            paused = self._wait_if_paused()
            if paused:
                paused_total += paused
                target_time += paused
                continue
            remaining = target_time - time.perf_counter()
            if remaining <= 0:
                break
            time.sleep(min(0.03, remaining))
        return paused_total

    def _put_latest(self, item: dict[str, Any]) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    return

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                if item["type"] == "frame":
                    self.status_var.set(item["status"])
                    self.stats_var.set(item["stats"])
                    self._update_preview(item["image"])
                elif item["type"] == "done":
                    self.status_var.set(item["status"])
                    self.pause_button.configure(text="Pause", state=tk.DISABLED)
                elif item["type"] == "error":
                    self.status_var.set("Error")
                    self.pause_button.configure(text="Pause", state=tk.DISABLED)
                    messagebox.showerror("Processing error", item["status"])
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def _update_preview(self, bgr_image) -> None:  # type: ignore[no-untyped-def]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        max_w = max(320, self.preview_label.winfo_width() - 12)
        max_h = max(240, self.preview_label.winfo_height() - 12)
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self.preview_image)


def main() -> None:
    root = tk.Tk()
    SocialBEVGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
