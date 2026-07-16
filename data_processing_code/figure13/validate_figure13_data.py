#!/usr/bin/env python3
"""Validate the self-contained Figure 13 inputs and outputs."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DERIVED = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", ROOT / "derived_data")).resolve() / "figure13"
RAW = ROOT / "raw_data" / "figure13"


def main() -> None:
    dense = pd.read_csv(DERIVED / "layer_fcv_dense_summary.csv")
    assert len(dense) == 21 * 50 * 4
    assert dense["epsilon"].nunique() == 21
    assert dense["run"].nunique() == 50
    assert dense["layer"].nunique() == 4
    assert dense["fcv_from_fc_state"].notna().all()
    assert (dense.groupby(["epsilon", "layer"]).size() == 50).all()

    large = pd.read_csv(RAW / "large_scale_fcv_observations.csv")
    counts = large.groupby("condition").size().to_dict()
    assert counts == {"Base": 145, "NULL-In": 125, "NULL-Out": 125}
    assert large["FCV"].notna().all()

    stats = pd.read_csv(ROOT / "statistics" / "figure13_stats_v2.csv")
    assert {"C", "E"}.issubset(set(stats["panel"].dropna()))
    image = Image.open(ROOT / "figures" / "figure13_final_v2.png")
    assert image.size == (4470, 3772)
    print("Validated Figure 13 layer-model, large-scale FCV, statistics, and image outputs")


if __name__ == "__main__":
    main()
