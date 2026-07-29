"""FCV/FCS forest plot using FCS-calibrated functional-unit SC measures.

This is deliberately separate from the historical Figure 12 forest plot.  It
uses the same legacy region ordering and rendering/statistical routines, but
only retains canonical regions with complete functional-unit structural data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_fcv_fcs_sc_corr_forest as forest


PACK_ROOT = Path(__file__).resolve().parents[1]
REGIONS_FILE = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
FUNCTIONAL_TABLE = PACK_ROOT / "derived_data" / "figure9" / "figure9_region_summary.csv"
DEFAULT_SC_SOURCE = "fcs_calibrated_endpoint"
FIGURE_BASENAME = "figure_fcv_fcs_fu_sc_corr_forest"
TWO_WAY_SUBSAMPLING_CSV = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "validation"
    / "two_way_subsampling_r12"
    / "figure_fcv_fcs_fu_sc_corr_forest_r12_two_way_subsampling_iterations.csv.gz"
)

# Keep the original forest layout, bootstrap, FDR procedure, and selected
# scatter panels while replacing only the structural feature set.
FU_SC_MEASURES = [
    ("Hard_OO_fraction", "OO frac."),
    ("FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("FU_DCApre", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    #("OutputParticipation", "Output part."),
    ("Reciprocity", "Reciprocity"),
    ("LogOutIn", "log(O/I)"),
]
FU_SCAT_COLS = [
    ("Hard_OO_fraction", "OO frac."),
    ("FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw FCV/FCS forest plot with functional-unit SC measures")
    parser.add_argument("--sc-source", default=DEFAULT_SC_SOURCE)
    parser.add_argument(
        "--panel-ab-distribution",
        choices=("bootstrap", "two_way_subsampling"),
        default="bootstrap",
        help="Distribution shown in forest panels A/B.",
    )
    parser.add_argument(
        "--subsampling-csv",
        type=Path,
        default=TWO_WAY_SUBSAMPLING_CSV,
        help="Iteration CSV used when --panel-ab-distribution=two_way_subsampling.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix added to output files. Defaults to the SC source for non-canonical sources.",
    )
    parser.add_argument(
        "--panel-ab-p-mode",
        choices=("fdr_bh", "raw"),
        default="fdr_bh",
        help="P value used for significance labels in panels A/B.",
    )
    return parser.parse_args()


def paths_for(sc_source: str, output_suffix: str) -> tuple[Path, str, Path, Path, Path]:
    structural_table = (
        PACK_ROOT
        / "derived_data"
        / "figure12"
        / "functional_unit_region_measures"
        / sc_source
        / "figure12_functional_unit_region_summary.csv"
    )
    suffix = output_suffix or ("" if sc_source == DEFAULT_SC_SOURCE else sc_source)
    figure_name = FIGURE_BASENAME if not suffix else f"{FIGURE_BASENAME}_{suffix}"
    matched_table = PACK_ROOT / "derived_data" / "common" / f"{figure_name}_input.csv"
    out_png = PACK_ROOT / "figures" / f"{figure_name}.png"
    out_stats = PACK_ROOT / "statistics" / f"{figure_name}_stats.csv"
    return structural_table, figure_name, matched_table, out_png, out_stats


def build_frame(structural_table: Path) -> tuple[pd.DataFrame, list[str]]:
    """Match FC and FU-SC summaries on the canonical region order."""
    regions = pd.read_csv(REGIONS_FILE)
    functional = pd.read_csv(FUNCTIONAL_TABLE)[
        ["root_area_id", "node", "EdgeStdFCV", "FCS"]
    ]
    sc_columns = [column for column, _ in FU_SC_MEASURES]
    structural = pd.read_csv(structural_table)[["root_area_id", "node", *sc_columns]]
    frame = (
        regions
        .merge(functional, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(structural, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
        .sort_values("legacy_order")
        .reset_index(drop=True)
    )
    measures = ["EdgeStdFCV", "FCS", *sc_columns]
    complete = frame[measures].notna().all(axis=1)
    excluded = frame.loc[~complete, "node"].astype(str).tolist()
    frame = frame.loc[complete].copy().reset_index(drop=True)
    if len(frame) < 4:
        raise RuntimeError("Too few complete matched regions for correlation analysis")
    for column in measures:
        frame[f"{column}_raw"] = frame[column]
        frame[column] = forest._zscore(frame[column].to_numpy(float))
    return frame, excluded


def _install_two_way_subsampling_distribution(samples_path: Path) -> None:
    samples = pd.read_csv(samples_path)
    required = {"func_column", "sc_column", "pearson_r"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"{samples_path} is missing columns: {sorted(missing)}")

    order = [
        (func_col, sc_col)
        for func_col, _, _ in forest.FUNC
        for sc_col, _ in forest.SC_MEASURES
    ]
    distributions = {}
    for func_col, sc_col in order:
        values = samples.loc[
            samples["func_column"].eq(func_col) & samples["sc_column"].eq(sc_col),
            "pearson_r",
        ].dropna().to_numpy(float)
        if len(values) < 10:
            raise ValueError(f"Too few subsampling values for {func_col} vs {sc_col}: {len(values)}")
        distributions[(func_col, sc_col)] = values

    call_index = {"value": 0}
    original_bootstrap = forest.bootstrap_pearson

    def two_way_subsampling_pearson(x, y):
        key = order[call_index["value"]]
        call_index["value"] += 1
        result = original_bootstrap(x, y)
        dist = distributions[key]
        result["boot"] = dist
        result["lo"], result["hi"] = [float(v) for v in np.percentile(dist, [2.5, 97.5])]
        return result

    forest.bootstrap_pearson = two_way_subsampling_pearson
    forest.N_BOOT = int(min(len(values) for values in distributions.values()))


def main() -> None:
    args = _parse_args()
    if args.panel_ab_distribution == "two_way_subsampling" and not args.output_suffix:
        args.output_suffix = f"{args.sc_source}_two_way_subsampling"
    structural_table, figure_name, matched_table, out_png, out_stats = paths_for(
        args.sc_source,
        args.output_suffix,
    )
    # The renderer reads these module-level definitions at call time.
    forest.SC_MEASURES = FU_SC_MEASURES
    forest.SCAT_COLS = FU_SCAT_COLS
    forest.PANEL_AB_P_MODE = args.panel_ab_p_mode
    if args.panel_ab_distribution == "two_way_subsampling":
        _install_two_way_subsampling_distribution(args.subsampling_csv)

    frame, excluded = build_frame(structural_table)
    matched_table.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(matched_table, index=False)

    stats_rows = []
    figure = forest.make_figure(frame, stats_rows)
    figure.savefig(out_png, dpi=600, bbox_inches="tight", transparent=False)
    plt.close(figure)

    stats = pd.DataFrame(stats_rows)
    stats["figure"] = figure_name
    stats["SC_source"] = args.sc_source
    stats["panel_ab_distribution"] = args.panel_ab_distribution
    stats["panel_ab_p_mode"] = args.panel_ab_p_mode
    stats["region_subset"] = "legacy_42_regions_no_rOB_complete_functional_unit_SC"
    stats["n_complete_regions"] = len(frame)
    stats["excluded_missing_fu_sc"] = ",".join(excluded) if excluded else ""
    stats.to_csv(out_stats, index=False)

    print(f"n = {len(frame)} complete regions; excluded: {excluded or 'none'}")
    print(f"Saved {matched_table}")
    print(f"Saved {out_png}")
    print(f"Saved {out_stats}")


if __name__ == "__main__":
    main()
