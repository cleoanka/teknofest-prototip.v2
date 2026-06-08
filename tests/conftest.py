"""Pytest ortak fixture'ları + repo kökünü sys.path'e ekler."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aura.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    """Varsayılan config (config/default.yaml)."""
    return load_config()
