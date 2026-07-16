#!/usr/bin/env python3
"""Merge the independently prepared FC and TE inputs for Figure 9."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DERIVED_DIR, PACK_ROOT, ensure_output_dirs

import sys

if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import anatomy_group


def main() -> None:
    ensure_output_dirs()
    fc = pd.read_csv(DERIVED_DIR / "figure9_fc_measures_recording_region.csv")
    te = pd.read_csv(DERIVED_DIR / "figure9_te_measures_recording_region.csv")
    if fc.duplicated(["recording_id", "node"]).any():
        raise RuntimeError("Duplicate recording-region keys in FC table")
    if te.duplicated(["recording_id", "node"]).any():
        raise RuntimeError("Duplicate recording-region keys in TE table")

    te_values = te[["recording_id", "node", "NetTE", "NeighborNetTE", "te_source"]].copy()
    table = fc.merge(te_values, on=["recording_id", "node"], how="left", validate="one_to_one")
    table.insert(0, "species", "Zebrafish")
    table["anatomy_group"] = table["node"].map(anatomy_group)
    table["complete_five_measures"] = table[
        ["EdgeStdFCV", "FCS", "ProfileCorrDistFCV", "NetTE", "NeighborNetTE"]
    ].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    table.to_csv(DERIVED_DIR / "figure9_recording_region_measures.csv", index=False)

    summary = (
        table.groupby(["root_area_id", "node", "anatomy_group"], as_index=False)
        .agg(
            n_recordings=("recording_id", "nunique"),
            n_clusters_mean=("n_clusters", "mean"),
            EdgeStdFCV=("EdgeStdFCV", "mean"),
            FCS=("FCS", "mean"),
            ProfileCorrDistFCV=("ProfileCorrDistFCV", "mean"),
            NetTE=("NetTE", "mean"),
            NeighborNetTE=("NeighborNetTE", "mean"),
        )
    )
    summary["complete_five_measures"] = summary[
        ["EdgeStdFCV", "FCS", "ProfileCorrDistFCV", "NetTE", "NeighborNetTE"]
    ].notna().all(axis=1)
    summary.to_csv(DERIVED_DIR / "figure9_region_summary.csv", index=False)

    qc = pd.DataFrame(
        [
            {"quantity": "recording_region_rows", "value": len(table)},
            {"quantity": "recordings", "value": table["recording_id"].nunique()},
            {"quantity": "fc_regions", "value": table["node"].nunique()},
            {"quantity": "complete_five_measure_regions", "value": int(summary["complete_five_measures"].sum())},
            {"quantity": "missing_nette_rows", "value": int(table["NetTE"].isna().sum())},
            {"quantity": "missing_neighbor_nette_rows", "value": int(table["NeighborNetTE"].isna().sum())},
        ]
    )
    qc.to_csv(DERIVED_DIR / "figure9_qc.csv", index=False)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
