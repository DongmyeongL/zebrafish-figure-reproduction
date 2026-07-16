#!/usr/bin/env python3
"""Build the Figure 12 subject-region and region-summary tables."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import DERIVED_DIR, PACK_ROOT, ensure_output_dirs

if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import anatomy_group


MEASURES = ["OO_fraction", "PostDCA", "PreDCA", "Modularity", "LogOutIn"]


def main() -> None:
    ensure_output_dirs()
    structural = pd.read_csv(DERIVED_DIR / "figure12_subject_region_structural_measures_all72.csv")
    figure9 = pd.read_csv(DERIVED_DIR.parent / "figure9" / "figure9_recording_region_measures.csv")
    keys = figure9[["recording_id", "root_area_id", "node"]].drop_duplicates()
    if keys.duplicated(["recording_id", "node"]).any():
        raise RuntimeError("Duplicate Figure 9 recording-region keys")
    table = keys.merge(
        structural,
        on=["recording_id", "node"],
        how="left",
        validate="one_to_one",
    )
    table.insert(0, "species", "Zebrafish")
    table["anatomy_group"] = table["node"].map(anatomy_group)
    table["complete_five_measures"] = table[MEASURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    table.to_csv(DERIVED_DIR / "figure12_subject_region_structural_measures.csv", index=False)

    aggregations = {measure: (measure, "mean") for measure in MEASURES}
    summary = (
        table.groupby(["root_area_id", "node", "anatomy_group"], as_index=False)
        .agg(n_subjects=("Subject", "nunique"), **aggregations)
    )
    summary["complete_five_measures"] = summary[MEASURES].notna().all(axis=1)
    summary.to_csv(DERIVED_DIR / "figure12_region_summary.csv", index=False)

    qc = pd.DataFrame(
        [
            {"quantity": "subject_region_rows", "value": len(table)},
            {"quantity": "subjects", "value": table["Subject"].nunique()},
            {"quantity": "regions", "value": table["node"].nunique()},
            {"quantity": "complete_subject_region_rows", "value": int(table["complete_five_measures"].sum())},
            {"quantity": "complete_region_means", "value": int(summary["complete_five_measures"].sum())},
            *[
                {"quantity": f"missing_{measure}_rows", "value": int(table[measure].isna().sum())}
                for measure in MEASURES
            ],
        ]
    )
    qc.to_csv(DERIVED_DIR / "figure12_qc.csv", index=False)
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
