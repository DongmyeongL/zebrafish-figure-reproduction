#!/usr/bin/env python3
"""Two-way subject/region subsampling robustness for the r12 FU-SC forest result."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PACK_ROOT = Path(__file__).resolve().parents[3]
if str(PACK_ROOT / "figure_code") not in sys.path:
    sys.path.insert(0, str(PACK_ROOT / "figure_code"))

import figure_style as fs


SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"
FC_TABLE = PACK_ROOT / "derived_data" / "figure9" / "figure9_recording_region_measures.csv"
SC_TABLE = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "functional_unit_region_measures"
    / SC_SOURCE
    / "figure12_subject_region_functional_unit_structural_measures.csv"
)
REGION_TABLE = PACK_ROOT / "derived_data" / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"
OUTPUT_DIR = PACK_ROOT / "derived_data" / "figure12" / "validation" / "two_way_subsampling_r12"
FIGURE_DIR = PACK_ROOT / "figures"

FUNCTIONAL_MEASURES = [
    ("EdgeStdFCV", "FCV"),
    ("FCS", "FCS"),
]
SC_MEASURES = [
    ("Hard_OO_fraction", "OO frac."),
    ("FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("FU_DCApre", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    ("Reciprocity", "Reciprocity"),
    ("LogOutIn", "log(O/I)"),
]


def load_matched_table() -> pd.DataFrame:
    regions = pd.read_csv(REGION_TABLE)[["root_area_id", "node", "legacy_order"]]
    fc = pd.read_csv(FC_TABLE)[
        ["recording_id", "root_area_id", "node", "EdgeStdFCV", "FCS", "anatomy_group"]
    ]
    sc = pd.read_csv(SC_TABLE)[
        [
            "recording_id",
            "root_area_id",
            "node",
            "Subject",
            "FU_DCApost",
            "FU_DCApre",
            "Hard_OO_fraction",
            "Reciprocity",
            "LogOutIn",
        ]
    ]
    table = (
        regions.merge(fc, on=["root_area_id", "node"], how="left")
        .merge(sc, on=["recording_id", "root_area_id", "node"], how="left")
        .replace([np.inf, -np.inf], np.nan)
    )
    return table.sort_values(["legacy_order", "recording_id"]).reset_index(drop=True)


def region_mean_frame(table: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col, _ in FUNCTIONAL_MEASURES] + [col for col, _ in SC_MEASURES]
    return (
        table.groupby(["root_area_id", "node", "legacy_order", "anatomy_group"], as_index=False)
        .agg(**{col: (col, "mean") for col in columns})
        .sort_values("legacy_order")
        .reset_index(drop=True)
    )


def correlations(region_means: pd.DataFrame) -> list[dict]:
    rows = []
    for func_col, func_label in FUNCTIONAL_MEASURES:
        for sc_col, sc_label in SC_MEASURES:
            pair = region_means[[func_col, sc_col]].dropna()
            if len(pair) < 4:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r, pearson_p = pearsonr(pair[func_col], pair[sc_col])
                spearman_r, spearman_p = spearmanr(pair[func_col], pair[sc_col])
            rows.append(
                {
                    "func": func_label,
                    "func_column": func_col,
                    "sc": sc_label,
                    "sc_column": sc_col,
                    "n_regions": int(len(pair)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }
            )
    return rows


def run_subsampling(table: pd.DataFrame, n_iter: int, region_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(table["Subject"].dropna().astype(int).unique()))
    regions = np.array(sorted(table["node"].astype(str).unique()))
    n_remove_regions = max(1, int(np.floor(len(regions) * region_fraction)))
    reference = pd.DataFrame(correlations(region_mean_frame(table)))
    rows = []
    for iteration in range(n_iter):
        removed_subject = int(rng.choice(subjects))
        removed_regions = rng.choice(regions, size=n_remove_regions, replace=False)
        keep = (
            table["Subject"].ne(removed_subject)
            & ~table["node"].astype(str).isin(set(removed_regions.astype(str)))
        )
        means = region_mean_frame(table.loc[keep].copy())
        for row in correlations(means):
            row.update(
                {
                    "iteration": iteration,
                    "removed_subject": removed_subject,
                    "removed_regions": ";".join(sorted(removed_regions.astype(str))),
                    "n_subjects_used": int(len(subjects) - 1),
                    "n_regions_removed": int(n_remove_regions),
                    "n_regions_available": int(len(means)),
                    "region_remove_fraction": float(region_fraction),
                    "seed": int(seed),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows), reference


def summarize(samples: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    ref = reference.set_index(["func_column", "sc_column"])
    for keys, group in samples.groupby(["func", "func_column", "sc", "sc_column"], sort=False):
        func, func_col, sc, sc_col = keys
        r = group["pearson_r"].dropna().to_numpy(float)
        rho = group["spearman_rho"].dropna().to_numpy(float)
        ref_row = ref.loc[(func_col, sc_col)]
        full_r = float(ref_row["pearson_r"])
        summary_rows.append(
            {
                "func": func,
                "func_column": func_col,
                "sc": sc,
                "sc_column": sc_col,
                "n_iterations": int(len(group)),
                "full_data_pearson_r": full_r,
                "full_data_pearson_p": float(ref_row["pearson_p"]),
                "subsample_median_pearson_r": float(np.nanmedian(r)),
                "subsample_ci2p5_pearson_r": float(np.nanpercentile(r, 2.5)),
                "subsample_ci97p5_pearson_r": float(np.nanpercentile(r, 97.5)),
                "fraction_pearson_r_positive": float(np.nanmean(r > 0)),
                "fraction_pearson_r_same_sign_as_full": float(np.nanmean(np.sign(r) == np.sign(full_r))),
                "fraction_pearson_p_lt_0p05": float(np.nanmean(group["pearson_p"].to_numpy(float) < 0.05)),
                "full_data_spearman_rho": float(ref_row["spearman_rho"]),
                "full_data_spearman_p": float(ref_row["spearman_p"]),
                "subsample_median_spearman_rho": float(np.nanmedian(rho)),
                "subsample_ci2p5_spearman_rho": float(np.nanpercentile(rho, 2.5)),
                "subsample_ci97p5_spearman_rho": float(np.nanpercentile(rho, 97.5)),
            }
        )
    return pd.DataFrame(summary_rows)


def draw_figure(samples: pd.DataFrame, reference: pd.DataFrame, output: Path) -> None:
    fs.apply_main_figure_style()
    ref = reference.set_index(["func_column", "sc_column"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    colors = {"FCV": "#2F6BBD", "FCS": "#8E4B9A"}
    for ax, func_label in zip(axes, ["FCV", "FCS"]):
        data = []
        labels = []
        full_rs = []
        for sc_col, sc_label in SC_MEASURES:
            subset = samples.loc[
                samples["func"].eq(func_label) & samples["sc_column"].eq(sc_col),
                "pearson_r",
            ].dropna().to_numpy(float)
            data.append(subset)
            labels.append(sc_label)
            ref_row = ref.loc[("EdgeStdFCV" if func_label == "FCV" else "FCS", sc_col)]
            full_rs.append(float(ref_row["pearson_r"]))
        viol = ax.violinplot(data, positions=np.arange(len(data)), widths=0.72, showextrema=False)
        for body in viol["bodies"]:
            body.set_facecolor(colors[func_label])
            body.set_edgecolor("#222222")
            body.set_alpha(0.32)
            body.set_linewidth(0.8)
        bp = ax.boxplot(
            data,
            positions=np.arange(len(data)),
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )
        ax.scatter(np.arange(len(data)), full_rs, s=42, c="white", edgecolors="black", zorder=5)
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"{func_label} Pearson r")
        ax.set_ylim(-0.85, 0.85)
        ax.set_title(f"{func_label}: remove one subject and 20% of regions per iteration", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xticks(np.arange(len(SC_MEASURES)))
    axes[-1].set_xticklabels(labels)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=5000)
    parser.add_argument("--region-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    table = load_matched_table()
    samples, reference = run_subsampling(table, args.n_iter, args.region_fraction, args.seed)
    summary = summarize(samples, reference)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUTPUT_DIR / "figure_fcv_fcs_fu_sc_corr_forest_r12_two_way_subsampling_iterations.csv"
    summary_path = OUTPUT_DIR / "figure_fcv_fcs_fu_sc_corr_forest_r12_two_way_subsampling_summary.csv"
    reference_path = OUTPUT_DIR / "figure_fcv_fcs_fu_sc_corr_forest_r12_full_data_reference.csv"
    figure_path = FIGURE_DIR / "figure_fcv_fcs_fu_sc_corr_forest_r12_two_way_subsampling_robustness.png"
    samples.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    reference.to_csv(reference_path, index=False)
    draw_figure(samples, reference, figure_path)

    print(f"Saved {sample_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {reference_path}")
    print(f"Saved {figure_path}")
    print(summary[["func", "sc", "full_data_pearson_r", "subsample_median_pearson_r", "subsample_ci2p5_pearson_r", "subsample_ci97p5_pearson_r", "fraction_pearson_r_same_sign_as_full"]].to_string(index=False))


if __name__ == "__main__":
    main()
