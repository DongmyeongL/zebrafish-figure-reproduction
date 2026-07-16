"""Shared paths and configuration for the clean Figure 12 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PACK_ROOT = Path(__file__).resolve().parents[2]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.pipeline_paths import derived_group, raw_group

CONFIG = json.loads((PACK_ROOT / "config" / "figure12.json").read_text())
RAW_DIR = raw_group("figure12")
DERIVED_DIR = derived_group("figure12")
CELL_DCA_DIR = DERIVED_DIR / "cell_dca"
ORIGINAL_SC_DIR = raw_group("figure9") / "original_subjects"

LREGION = [
    "MON", "Cb", "MOS1", "MOS2", "MOS3", "MOS4", "MOS5", "IPN", "IO", "Hc", "Ra", "T",
    "aRF", "imRF", "pRF", "GG", "Hb", "Hi", "HR", "OG", "OB", "OE", "P", "Pi", "PT",
    "PO", "PrT", "R", "SP", "TeO", "Th", "TL", "TS", "TG", "VR", "NX",
]
RREGION = [
    "rMON", "rCb", "rMOS1", "rMOS2", "rMOS3", "rMOS4", "rMOS5", "rIPN", "rIO", "rHc",
    "rRa", "rT", "raRF", "rimRF", "rpRF", "rGG", "rHb", "rHi", "rHR", "rOG", "rOB",
    "rOE", "rP", "rPi", "rPT", "rPO", "rPrT", "rR", "rSP", "rTeO", "rTh", "rTL",
    "rTS", "rTG", "rVR", "rNX",
]
REGION_NAMES = LREGION + RREGION
N_REGIONS = len(REGION_NAMES)


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    CELL_DCA_DIR.mkdir(parents=True, exist_ok=True)


def original_sc_path(subject: int) -> Path:
    return ORIGINAL_SC_DIR / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"


def compact_sc_path(subject: int) -> Path:
    return RAW_DIR / f"subject_{subject}_compact_sc.npz"


def cell_dca_path(subject: int) -> Path:
    return CELL_DCA_DIR / f"subject_{subject}_cell_dca.npz"
