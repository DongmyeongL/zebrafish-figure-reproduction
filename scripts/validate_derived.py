#!/usr/bin/env python3
"""Validate public raw-to-derived outputs without private reference files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


COMPARISONS = {
    "figure9": (
        "figure9/figure9_recording_region_measures.csv",
        ["recording_id", "node"],
        ["EdgeStdFCV", "FCS", "ProfileCorrDistFCV", "NetTE", "NeighborNetTE"],
    ),
    "figure12": (
        "figure12/figure12_subject_region_structural_measures.csv",
        ["Subject", "node"],
        ["OO_fraction", "PostDCA", "PreDCA", "Modularity", "LogOutIn"],
    ),
    "stimulus": (
        "figure_stimulus/stimulus_fc_measures_subject_condition_region.csv",
        ["subject", "stimulus_label", "node"],
        ["FCV_raw", "FCS_raw", "FCV_z", "FCS_z"],
    ),
    "celegans": (
        "invertebrates/celegans_node_metrics.csv",
        ["node"],
        ["EdgeStdFCV", "PostDCA", "PreDCA", "OO_fraction"],
    ),
    "drosophila": (
        "invertebrates/drosophila_region_metrics.csv",
        ["node"],
        ["EdgeStdFCV", "PostDCA", "PreDCA", "OO_fraction"],
    ),
}


def require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}")


def validate_figure9(derived: Path) -> None:
    table = pd.read_csv(derived / "figure9" / "figure9_recording_region_measures.csv")
    measures = ["EdgeStdFCV", "FCS", "ProfileCorrDistFCV", "NetTE", "NeighborNetTE"]
    require_columns(table, ["recording_id", "root_area_id", "node", *measures], "figure9")
    if table["recording_id"].nunique() != 7 or table["node"].nunique() != 66:
        raise RuntimeError("figure9: expected seven recordings and 66 root areas")
    canonical = pd.read_csv(derived / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv")
    subset = table[table["node"].isin(canonical["node"])]
    if len(subset) != 294 or not subset.groupby("node")["recording_id"].nunique().eq(7).all():
        raise RuntimeError("figure9: canonical 42-region set is incomplete")
    print(f"OK figure9: {len(table)} recording-region rows, 66 regions")


def validate_figure12(derived: Path) -> None:
    table = pd.read_csv(derived / "figure12" / "figure12_subject_region_structural_measures.csv")
    measures = ["OO_fraction", "PostDCA", "PreDCA", "Modularity", "LogOutIn"]
    require_columns(table, ["Subject", "recording_id", "node", *measures], "figure12")
    if table["Subject"].nunique() != 7 or table["node"].nunique() != 66:
        raise RuntimeError("figure12: expected seven subjects and 66 root areas")
    print(f"OK figure12: {len(table)} subject-region rows, 66 regions")


def validate_stimulus(derived: Path) -> None:
    table = pd.read_csv(derived / "figure_stimulus" / "stimulus_fc_measures_subject_condition_region.csv")
    require_columns(table, ["subject", "stimulus_label", "node", "FCV_raw", "FCS_raw"], "stimulus")
    if table["subject"].nunique() != 7 or table["stimulus_label"].nunique() != 3 or table["node"].nunique() != 66:
        raise RuntimeError("stimulus: expected seven subjects, three conditions, and 66 regions")
    print(f"OK stimulus: {len(table)} subject-condition-region rows")


def validate_invertebrate(derived: Path, species: str) -> None:
    filename = "celegans_node_metrics.csv" if species == "celegans" else "drosophila_region_metrics.csv"
    table = pd.read_csv(derived / "invertebrates" / filename)
    require_columns(table, ["node", "EdgeStdFCV", "PostDCA", "PreDCA", "OO_fraction"], species)
    if len(table) < 8:
        raise RuntimeError(f"{species}: unexpectedly small matched table")
    print(f"OK {species}: {len(table)} matched nodes/regions")


def validate_layer(derived: Path) -> None:
    dense = pd.read_csv(derived / "figure13" / "layer_fcv_dense_summary.csv")
    require_columns(dense, ["epsilon", "run", "layer", "fcv_from_fc_state"], "layer")
    if len(dense) != 21 * 50 * 4 or dense["fcv_from_fc_state"].isna().any():
        raise RuntimeError("layer: dense summary is incomplete")
    print("OK layer: 21 epsilon values x 50 runs x 4 layers")


def compare_with_bundled(target: str, derived: Path) -> None:
    if target not in COMPARISONS:
        print(f"SKIP comparison for {target}: no deterministic bundled-table check")
        return
    relative, keys, measures = COMPARISONS[target]
    current = pd.read_csv(derived / relative)
    reference = pd.read_csv(ROOT / "derived_data" / relative)
    merged = reference[keys + measures].merge(
        current[keys + measures],
        on=keys,
        how="outer",
        suffixes=("_reference", "_current"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{target}: key set differs from the bundled reference")
    for measure in measures:
        reference_values = pd.to_numeric(
            merged[f"{measure}_reference"], errors="coerce"
        ).to_numpy(float)
        current_values = pd.to_numeric(
            merged[f"{measure}_current"], errors="coerce"
        ).to_numpy(float)
        if not np.allclose(
            reference_values,
            current_values,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            finite = np.isfinite(reference_values) & np.isfinite(current_values)
            maximum = (
                float(np.max(np.abs(reference_values[finite] - current_values[finite])))
                if finite.any()
                else np.nan
            )
            raise RuntimeError(
                f"{target}: {measure} differs from bundled reference "
                f"(max absolute difference {maximum})"
            )
    print(f"OK {target}: deterministic measures match bundled reference")


VALIDATORS = {
    "figure9": validate_figure9,
    "figure12": validate_figure12,
    "stimulus": validate_stimulus,
    "celegans": lambda path: validate_invertebrate(path, "celegans"),
    "drosophila": lambda path: validate_invertebrate(path, "drosophila"),
    "layer": validate_layer,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=[*VALIDATORS, "all"])
    parser.add_argument("--derived-root", type=Path, default=ROOT / "derived_data")
    parser.add_argument(
        "--compare-bundled",
        action="store_true",
        help="Compare deterministic measures with the frozen tables in this release.",
    )
    args = parser.parse_args()
    targets = list(VALIDATORS) if args.target == "all" else [args.target]
    for target in targets:
        VALIDATORS[target](args.derived_root.resolve())
        if args.compare_bundled:
            compare_with_bundled(target, args.derived_root.resolve())


if __name__ == "__main__":
    main()
