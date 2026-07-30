"""Shared paths and configuration for the clean Figure 12 pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACK_ROOT.parents[1]
CONFIG = json.loads((PACK_ROOT / "config" / "figure12.json").read_text())
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK_ROOT / "raw_data"))
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data"))
RAW_DIR = RAW_ROOT / "figure12"
DERIVED_DIR = DERIVED_ROOT / "figure12"
CELL_DCA_DIR = DERIVED_DIR / "cell_dca"
FUNCTIONAL_UNIT_DCA_DIR = DERIVED_DIR / "functional_unit_dca"
FCS_CALIBRATED_ENDPOINT_SC_DIR = RAW_DIR / "fcs_calibrated_endpoint_sc"
FCS_CALIBRATED_SKELETON_PATH_SC_DIR = RAW_DIR / "fcs_calibrated_skeleton_path_sc"
FCS_CALIBRATED_SKELETON_NEAREST_ENDPOINT_SC_DIR = RAW_DIR / "fcs_calibrated_skeleton_nearest_endpoint_sc"
FCS_CALIBRATED_SKELETON_KMEANS_NEAREST_R11_SC_DIR = RAW_DIR / "fcs_calibrated_skeleton_kmeans_nearest_r11_sc"
FCS_CALIBRATED_SKELETON_KMEANS_NEAREST_R12_SC_DIR = RAW_DIR / "fcs_calibrated_skeleton_kmeans_nearest_r12_sc"
ORIGINAL_SC_DIR = Path(
    os.environ.get(
        "ZF_ORIGINAL_SC_DIR",
        WORKSPACE_ROOT
        / "fcv_postdca_raw_recompute"
        / "data"
        / "zebrafish"
        / "sc"
        / "original_raw_data",
    )
)

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

SC_SOURCE_CHOICES = (
    "historical",
    "fcs_calibrated_endpoint",
    "fcs_calibrated_skeleton_path",
    "fcs_calibrated_skeleton_nearest_endpoint",
    "fcs_calibrated_skeleton_kmeans_nearest_r11",
    "fcs_calibrated_skeleton_kmeans_nearest_r12",
)


def ensure_output_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    CELL_DCA_DIR.mkdir(parents=True, exist_ok=True)
    FUNCTIONAL_UNIT_DCA_DIR.mkdir(parents=True, exist_ok=True)


def original_sc_path(subject: int) -> Path:
    return ORIGINAL_SC_DIR / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"


def compact_sc_path(subject: int, sc_source: str = "historical") -> Path:
    if sc_source == "fcs_calibrated_endpoint":
        return FCS_CALIBRATED_ENDPOINT_SC_DIR / f"subject_{subject}_compact_sc.npz"
    if sc_source == "fcs_calibrated_skeleton_path":
        return FCS_CALIBRATED_SKELETON_PATH_SC_DIR / f"subject_{subject}_compact_sc.npz"
    if sc_source == "fcs_calibrated_skeleton_nearest_endpoint":
        return FCS_CALIBRATED_SKELETON_NEAREST_ENDPOINT_SC_DIR / f"subject_{subject}_compact_sc.npz"
    if sc_source == "fcs_calibrated_skeleton_kmeans_nearest_r11":
        return FCS_CALIBRATED_SKELETON_KMEANS_NEAREST_R11_SC_DIR / f"subject_{subject}_compact_sc.npz"
    if sc_source == "fcs_calibrated_skeleton_kmeans_nearest_r12":
        return FCS_CALIBRATED_SKELETON_KMEANS_NEAREST_R12_SC_DIR / f"subject_{subject}_compact_sc.npz"
    if sc_source != "historical":
        raise ValueError(f"Unknown Figure 12 SC source: {sc_source}")
    return RAW_DIR / f"subject_{subject}_compact_sc.npz"


def cell_dca_path(subject: int, sc_source: str = "historical") -> Path:
    if sc_source == "historical":
        return CELL_DCA_DIR / f"subject_{subject}_cell_dca.npz"
    path = CELL_DCA_DIR / sc_source
    path.mkdir(parents=True, exist_ok=True)
    return path / f"subject_{subject}_cell_dca.npz"


def functional_unit_dca_path(subject: int, sc_source: str) -> Path:
    path = FUNCTIONAL_UNIT_DCA_DIR / sc_source
    path.mkdir(parents=True, exist_ok=True)
    return path / f"subject_{subject}_functional_unit_dca.npz"
