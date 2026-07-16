from __future__ import annotations

import json
from pathlib import Path
import sys


PACK_ROOT = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.pipeline_paths import derived_group, raw_group

CONFIG = json.loads((PACK_ROOT / "config" / "figure9.json").read_text())
RAW_DIR = raw_group("figure9")
DERIVED_DIR = derived_group("figure9")


def ensure_output_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    (DERIVED_DIR / "functional_unit_traces").mkdir(parents=True, exist_ok=True)
