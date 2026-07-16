"""Environment-configurable paths shared by raw-to-derived pipelines."""

from __future__ import annotations

import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


RAW_DATA_ROOT = _configured_path("ZF_RAW_DATA_ROOT", RELEASE_ROOT / "raw_data")
DERIVED_DATA_ROOT = _configured_path(
    "ZF_DERIVED_DATA_ROOT", RELEASE_ROOT / "derived_data"
)


def raw_group(name: str) -> Path:
    return RAW_DATA_ROOT / name


def derived_group(name: str) -> Path:
    return DERIVED_DATA_ROOT / name
