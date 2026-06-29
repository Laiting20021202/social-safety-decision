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
    zone_config_dir: str = "config/zones"
