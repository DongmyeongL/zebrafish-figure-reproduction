#!/usr/bin/env python3
"""Build Figure 12 subject-region and region-summary tables.

The frozen cell-level analysis remains the default.  The functional-unit path
builds a separate table from the FCS-calibrated endpoint-derived SC, so that
the historical Figure 12 derived tables are never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import DERIVED_DIR, PACK_ROOT, SC_SOURCE_CHOICES, ensure_output_dirs

if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import anatomy_group


CELL_MEASURES = ["OO_fraction", "PostDCA", "PreDCA", "Modularity", "LogOutIn"]
FUNCTIONAL_UNIT_MEASURES = [
    "Hard_OO_fraction",
    "FU_DCApost",
    "FU_DCApre",
    "LogOutIn",
    "OutputParticipation",
    "Reciprocity",
    "CycleParticipation",
]
# Cycle participation is retained as an exploratory measure, but it is often
# saturated in dense endpoint-derived SCs and is therefore not a completeness
# requirement for the primary functional-unit table.
FUNCTIONAL_UNIT_PRIMARY_MEASURES = [
    "Hard_OO_fraction",
    "FU_DCApost",
    "FU_DCApre",
    "LogOutIn",
    "OutputParticipation",
    "Reciprocity",
]


def table_paths(analysis: str, sc_source: str) -> tuple[Path, Path, Path, Path, list[str], list[str]]:
    """Return source and output paths without mixing SC-analysis variants."""
    if analysis == "cell":
        if sc_source != "historical":
            raise ValueError("Cell-level Figure 12 tables support only the frozen historical SC")
        return (
            DERIVED_DIR / "figure12_subject_region_structural_measures_all72.csv",
            DERIVED_DIR / "figure12_subject_region_structural_measures.csv",
            DERIVED_DIR / "figure12_region_summary.csv",
            DERIVED_DIR / "figure12_qc.csv",
            CELL_MEASURES,
            CELL_MEASURES,
        )

    output_dir = DERIVED_DIR / "functional_unit_region_measures" / sc_source
    return (
        output_dir / "figure12_subject_region_functional_unit_structural_measures_all72.csv",
        output_dir / "figure12_subject_region_functional_unit_structural_measures.csv",
        output_dir / "figure12_functional_unit_region_summary.csv",
        output_dir / "figure12_functional_unit_qc.csv",
        FUNCTIONAL_UNIT_MEASURES,
        FUNCTIONAL_UNIT_PRIMARY_MEASURES,
    )


def main(analysis: str, sc_source: str) -> None:
    ensure_output_dirs()
    input_path, table_path, summary_path, qc_path, measures, primary_measures = table_paths(analysis, sc_source)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Figure 12 structural input: {input_path}")

    structural = pd.read_csv(input_path)
    missing_columns = sorted(set(measures) - set(structural.columns))
    if missing_columns:
        raise ValueError(f"{input_path.name} is missing expected columns: {missing_columns}")
    figure9 = pd.read_csv(PACK_ROOT / "derived_data" / "figure9" / "figure9_recording_region_measures.csv")
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
    table.insert(1, "SC_analysis", analysis)
    table.insert(2, "SC_source", sc_source)
    table["anatomy_group"] = table["node"].map(anatomy_group)
    finite = table[measures].replace([np.inf, -np.inf], np.nan)
    table[measures] = finite
    table["complete_primary_measures"] = table[primary_measures].notna().all(axis=1)
    if analysis == "cell":
        # Keep the legacy column name for backwards compatibility.
        table["complete_five_measures"] = table["complete_primary_measures"]
    table.to_csv(table_path, index=False)

    aggregations = {measure: (measure, "mean") for measure in measures}
    summary = (
        table.groupby(["root_area_id", "node", "anatomy_group"], as_index=False)
        .agg(n_subjects=("Subject", "nunique"), **aggregations)
    )
    summary["complete_primary_measures"] = summary[primary_measures].notna().all(axis=1)
    if analysis == "cell":
        summary["complete_five_measures"] = summary["complete_primary_measures"]
    summary.to_csv(summary_path, index=False)

    qc = pd.DataFrame(
        [
            {"quantity": "analysis", "value": analysis},
            {"quantity": "sc_source", "value": sc_source},
            {"quantity": "subject_region_rows", "value": len(table)},
            {"quantity": "subjects", "value": table["Subject"].nunique()},
            {"quantity": "regions", "value": table["node"].nunique()},
            {"quantity": "complete_primary_subject_region_rows", "value": int(table["complete_primary_measures"].sum())},
            {"quantity": "complete_primary_region_means", "value": int(summary["complete_primary_measures"].sum())},
            *[
                {"quantity": f"missing_{measure}_rows", "value": int(table[measure].isna().sum())}
                for measure in measures
            ],
        ]
    )
    qc.to_csv(qc_path, index=False)
    print(f"Saved {table_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {qc_path}")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("cell", "functional_unit"), default="functional_unit")
    parser.add_argument("--sc-source", choices=SC_SOURCE_CHOICES, default="fcs_calibrated_skeleton_kmeans_nearest_r12")
    args = parser.parse_args()
    main(args.analysis, args.sc_source)
