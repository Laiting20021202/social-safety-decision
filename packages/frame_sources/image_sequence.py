from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from packages.common_models import DatasetInfo, FramePacket, ScenarioInfo
from packages.frame_sources.base import FrameSource


class ImageSequenceSource(FrameSource):
    def __init__(
        self,
        root: str | Path,
        dataset_name: str = "local-image-sequence",
        revision: str = "local",
        split: str = "prompts",
        interval_sec: float = 0.5,
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.revision = revision
        self.split = split
        self.interval_sec = interval_sec

    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            dataset_id=self.dataset_name,
            name=self.dataset_name,
            revision=self.revision,
            source_url=str(self.root),
            cached=True,
            metadata={"source": "local"},
        )

    def list_scenarios(self) -> list[ScenarioInfo]:
        scenarios: list[ScenarioInfo] = []
        prompts_root = self.root / self.split
        if not prompts_root.exists():
            prompts_root = self.root
        for directory in sorted(path for path in prompts_root.iterdir() if path.is_dir()):
            frame_paths = self._frame_paths(directory.name)
            if not frame_paths:
                continue
            width, height = self._image_size(frame_paths[0])
            scenarios.append(
                ScenarioInfo(
                    scenario_id=directory.name,
                    dataset_name=self.dataset_name,
                    dataset_revision=self.revision,
                    split=self.split,
                    frame_count=len(frame_paths),
                    duration_sec=max(0.0, (len(frame_paths) - 1) * self.interval_sec),
                    image_width=width,
                    image_height=height,
                    first_frame_index=0,
                    last_frame_index=len(frame_paths) - 1,
                    metadata={"timestamp_source": "virtual_from_sequence"},
                )
            )
        return scenarios

    def get_scenario(self, scenario_id: str) -> ScenarioInfo:
        for scenario in self.list_scenarios():
            if scenario.scenario_id == scenario_id:
                return scenario
        raise FileNotFoundError(f"Scenario not found: {scenario_id}")

    def get_frame(self, scenario_id: str, frame_index: int) -> FramePacket:
        scenario = self.get_scenario(scenario_id)
        if frame_index < 0 or frame_index >= scenario.frame_count:
            raise IndexError(f"Frame {frame_index} outside scenario {scenario_id}")
        path = self.get_frame_image_path(scenario_id, frame_index)
        return FramePacket(
            source_type="image_sequence",
            dataset_name=self.dataset_name,
            dataset_revision=self.revision,
            split=self.split,
            scenario_id=scenario_id,
            frame_index=frame_index,
            timestamp_sec=frame_index * self.interval_sec,
            original_timestamp=None,
            fps=None,
            image_width=scenario.image_width,
            image_height=scenario.image_height,
            image_reference=f"/scenarios/{scenario_id}/frames/{frame_index}/image",
            metadata={
                "path": str(path),
                "timestamp_source": "virtual_from_sequence",
                "virtual_frame_interval_sec": self.interval_sec,
            },
        )

    def get_frame_image_path(self, scenario_id: str, frame_index: int) -> Path:
        paths = self._frame_paths(scenario_id)
        if frame_index < 0 or frame_index >= len(paths):
            raise IndexError(f"Frame {frame_index} outside scenario {scenario_id}")
        return paths[frame_index]

    def _scenario_dir(self, scenario_id: str) -> Path:
        prompts_path = self.root / self.split / scenario_id
        if prompts_path.exists():
            return prompts_path
        return self.root / scenario_id

    def _frame_paths(self, scenario_id: str) -> list[Path]:
        directory = self._scenario_dir(scenario_id)
        if not directory.exists():
            return []
        paths = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        return sorted(paths, key=_natural_frame_key)

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)


def _natural_frame_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return (path.name, int(match.group(1)) if match else -1)
