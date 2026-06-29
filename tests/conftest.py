from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def fixture_socialnav_root() -> Path:
    return Path(__file__).parent / "fixtures" / "socialnav_sub"
