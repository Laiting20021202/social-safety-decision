from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from packages.frame_sources.socialnav_sub import DEFAULT_DATASET_ID, DEFAULT_DATASET_REVISION


class DatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_mode: str = "dataset_playback"
    socialnav_dataset_id: str = DEFAULT_DATASET_ID
    socialnav_dataset_revision: str = DEFAULT_DATASET_REVISION
    socialnav_cache_dir: str | None = None
    socialnav_local_repo: str | None = None
    socialnav_virtual_frame_interval_sec: float = 0.5
    dataset_cache_dir: str = "outputs/dataset_cache"
    sam3_analysis_width: int = 640
    sam3_analysis_height: int = 360
    sam3_tracking_fps: float = 5.0
    sam3_max_objects: int = 6
    zone_config_dir: str = "config/zones"
