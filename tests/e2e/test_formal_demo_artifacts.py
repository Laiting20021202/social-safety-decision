from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.model_access, pytest.mark.slow]


def test_formal_e2e_demo_artifacts_exist() -> None:
    if os.getenv("RUN_FORMAL_GPU_TESTS") != "1":
        pytest.skip("Set RUN_FORMAL_GPU_TESTS=1 after real SAM 3/SAM 3.1 access is available.")

    root = Path("outputs/e2e_demo")
    required = [
        root / "manifest.json",
        root / "analysis.jsonl",
        root / "demo_overlay.mp4",
        root / "report.md",
        root / "screenshots",
        root / "browser_logs",
        root / "service_logs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Formal E2E artifacts are missing: {missing}"
