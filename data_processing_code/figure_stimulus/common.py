from __future__ import annotations

import json
from pathlib import Path
import sys


PACK_ROOT = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.pipeline_paths import DERIVED_DATA_ROOT, raw_group

CONFIG = json.loads((PACK_ROOT / "config" / "figure_stimulus.json").read_text())
SOURCE_DIR = raw_group("figure9") / "original_subjects"
LABELS_FILE = raw_group("figure9") / "region_labels.csv"
ANALYSIS_REGIONS_FILE = DERIVED_DATA_ROOT / "figure9" / "figure9_region_summary.csv"
DERIVED_DIR = DERIVED_DATA_ROOT / "figure_stimulus"


def ensure_output_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    (DERIVED_DIR / "region_traces").mkdir(parents=True, exist_ok=True)
