from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from packages.common_models import DatasetInfo, FramePacket, ScenarioInfo
from packages.frame_sources.base import FrameSource

DEFAULT_DATASET_ID = "michaelmunje/SocialNav-SUB"
DEFAULT_DATASET_REVISION = "f750caf46e5b33e6aef8c95af6a92fb4aff1d1b1"


class HuggingFaceDatasetSource(FrameSource):
    def __init__(
        self,
        dataset_id: str = DEFAULT_DATASET_ID,
        revision: str = DEFAULT_DATASET_REVISION,
        split: str = "prompts",
        cache_dir: str | Path | None = None,
        local_repo: str | Path | None = None,
        virtual_frame_interval_sec: float = 0.5,
        token: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.revision = revision
        self.split = split
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        local_value = local_repo or os.getenv("SOCIALNAV_LOCAL_REPO")
        self.local_repo = Path(local_value).expanduser() if local_value else None
        self.virtual_frame_interval_sec = virtual_frame_interval_sec
        self.token = token or os.getenv("HF_TOKEN") or None
        self._api = HfApi(token=self.token)

    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            dataset_id=self.dataset_id,
            name="SocialNav-SUB",
            revision=self.revision,
            license="MIT",
            source_url=f"https://huggingface.co/datasets/{self.dataset_id}",
            cached=bool(self.local_repo),
            metadata={
                "adapter": "HuggingFaceDatasetSource",
                "local_repo": str(self.local_repo) if self.local_repo else None,
            },
        )

    def list_scenarios(self) -> list[ScenarioInfo]:
        scenarios: list[ScenarioInfo] = []
        for scenario_id, frame_files in sorted(self._scenario_frame_files().items()):
            frame_count = len(frame_files)
            scenarios.append(
                ScenarioInfo(
                    scenario_id=scenario_id,
                    dataset_name=self.dataset_id,
                    dataset_revision=self.revision,
                    split=self.split,
                    frame_count=frame_count,
                    duration_sec=max(0.0, (frame_count - 1) * self.virtual_frame_interval_sec),
                    image_width=0,
                    image_height=0,
                    first_frame_index=0,
                    last_frame_index=frame_count - 1,
                    metadata={
                        "timestamp_source": "virtual_from_sequence",
                        "source_directory": f"{self.split}/{scenario_id}",
                        "dimensions_loaded": False,
                    },
                )
            )
        return scenarios

    def get_scenario(self, scenario_id: str) -> ScenarioInfo:
        frame_files = self._scenario_frame_files().get(scenario_id)
        if not frame_files:
            raise FileNotFoundError(f"Scenario not found: {scenario_id}")
        first_path = self._materialize_file(frame_files[0])
        width, height = self._image_size(first_path)
        frame_count = len(frame_files)
        return ScenarioInfo(
            scenario_id=scenario_id,
            dataset_name=self.dataset_id,
            dataset_revision=self.revision,
            split=self.split,
            frame_count=frame_count,
            duration_sec=max(0.0, (frame_count - 1) * self.virtual_frame_interval_sec),
            image_width=width,
            image_height=height,
            first_frame_index=0,
            last_frame_index=frame_count - 1,
            metadata={
                "timestamp_source": "virtual_from_sequence",
                "source_directory": f"{self.split}/{scenario_id}",
            },
        )

    def get_frame(self, scenario_id: str, frame_index: int) -> FramePacket:
        scenario = self.get_scenario(scenario_id)
        if frame_index < 0 or frame_index >= scenario.frame_count:
            raise IndexError(f"Frame {frame_index} outside scenario {scenario_id}")
        local_path = self.get_frame_image_path(scenario_id, frame_index)
        return FramePacket(
            source_type="huggingface",
            dataset_name=self.dataset_id,
            dataset_revision=self.revision,
            split=self.split,
            scenario_id=scenario_id,
            frame_index=frame_index,
            timestamp_sec=frame_index * self.virtual_frame_interval_sec,
            original_timestamp=None,
            fps=None,
            image_width=scenario.image_width,
            image_height=scenario.image_height,
            image_reference=f"/scenarios/{scenario_id}/frames/{frame_index}/image",
            metadata={
                "path": str(local_path),
                "timestamp_source": "virtual_from_sequence",
                "virtual_frame_interval_sec": self.virtual_frame_interval_sec,
                "source_file": self._scenario_frame_files()[scenario_id][frame_index],
            },
        )

    def get_frame_image_path(self, scenario_id: str, frame_index: int) -> Path:
        frame_files = self._scenario_frame_files().get(scenario_id)
        if not frame_files:
            raise FileNotFoundError(f"Scenario not found: {scenario_id}")
        if frame_index < 0 or frame_index >= len(frame_files):
            raise IndexError(f"Frame {frame_index} outside scenario {scenario_id}")
        return self._materialize_file(frame_files[frame_index])

    @cached_property
    def _repo_files(self) -> list[str]:
        if self.local_repo:
            return self._local_repo_files(self.local_repo)
        return list(
            self._api.list_repo_files(
                repo_id=self.dataset_id,
                repo_type="dataset",
                revision=self.revision,
            )
        )

    def _scenario_frame_files(self) -> dict[str, list[str]]:
        scenarios: dict[str, list[str]] = {}
        prefix = f"{self.split}/"
        for filename in self._repo_files:
            if not filename.startswith(prefix):
                continue
            parts = filename.split("/")
            if len(parts) != 3:
                continue
            scenario_id = parts[1]
            basename = parts[2]
            if not _is_frame_file(basename):
                continue
            scenarios.setdefault(scenario_id, []).append(filename)
        for files in scenarios.values():
            files.sort(key=_natural_frame_key)
        return scenarios

    def _materialize_file(self, repo_file: str) -> Path:
        if self.local_repo:
            return self.local_repo / repo_file
        path = hf_hub_download(
            repo_id=self.dataset_id,
            repo_type="dataset",
            revision=self.revision,
            filename=repo_file,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            token=self.token,
        )
        return Path(path)

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)

    @staticmethod
    def _local_repo_files(root: Path) -> list[str]:
        return [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.parts)
        ]


def _is_frame_file(filename: str) -> bool:
    lowered = filename.lower()
    return lowered.startswith("sample_with_bev_") and lowered.endswith((".png", ".jpg", ".jpeg"))


def _natural_frame_key(path_or_name: str | Path) -> tuple[str, int]:
    name = Path(path_or_name).name
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return (name, int(match.group(1)) if match else -1)


def scenario_ids_from_files(files: Iterable[str], split: str = "prompts") -> list[str]:
    prefix = f"{split}/"
    ids = set()
    for filename in files:
        if filename.startswith(prefix):
            parts = filename.split("/")
            if len(parts) >= 3 and _is_frame_file(parts[-1]):
                ids.add(parts[1])
    return sorted(ids)
