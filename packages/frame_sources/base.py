from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from packages.common_models import DatasetInfo, FramePacket, ScenarioInfo


class FrameSource(ABC):
    @abstractmethod
    def dataset_info(self) -> DatasetInfo:
        """Return dataset metadata."""

    @abstractmethod
    def list_scenarios(self) -> list[ScenarioInfo]:
        """Return available scenarios."""

    @abstractmethod
    def get_scenario(self, scenario_id: str) -> ScenarioInfo:
        """Return one scenario."""

    @abstractmethod
    def get_frame(self, scenario_id: str, frame_index: int) -> FramePacket:
        """Return frame metadata."""

    @abstractmethod
    def get_frame_image_path(self, scenario_id: str, frame_index: int) -> Path:
        """Return a local image path, downloading or materializing it if needed."""
