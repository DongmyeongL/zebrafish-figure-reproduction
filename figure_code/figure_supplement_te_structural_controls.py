#!/usr/bin/env python3
"""TE and skeleton-r12 directed structural controls from clean Figure 9/12 tables."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr
try:
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
    def multipletests(pvals, alpha=0.05, method="holm"):
        pvals = np.asarray(pvals, dtype=float)
        n = len(pvals)
        order = np.argsort(pvals)
        corrected = np.full(n, np.nan, dtype=float)
        if method == "holm":
            adjusted_sorted = np.empty(n, dtype=float)
            running = 0.0
            for rank, idx in enumerate(order):
                value = (n - rank) * pvals[idx]
                running = max(running, value)
                adjusted_sorted[rank] = min(running, 1.0)
            corrected[order] = adjusted_sorted
        elif method in {"fdr_bh", "bh"}:
            adjusted_sorted = np.empty(n, dtype=float)
            running = 1.0
            for rank_from_end, idx in enumerate(order[::-1], start=1):
                rank = n - rank_from_end + 1
                value = pvals[idx] * n / rank
                running = min(running, value)
                adjusted_sorted[n - rank_from_end] = min(running, 1.0)
            corrected[order] = adjusted_sorted
        else:
            raise ValueError(f"Unsupported correction method without statsmodels: {method}")
        reject = corrected <= alpha
        return reject, corrected, None, None


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import ANATOMY_GROUP_ORDER

import figure_style as fs
import figure_fcv_fcs_sc_corr_forest as forest


FUNCTIONAL_TABLE = (
    PACK_ROOT / "derived_data" / "figure9" / "figure9_recording_region_measures.csv"
)
STRUCTURAL_TABLE = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "functional_unit_region_measures"
    / "fcs_calibrated_skeleton_kmeans_nearest_r12"
    / "figure12_subject_region_functional_unit_structural_measures.csv"
)
SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"
REGIONS_FILE = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
MAIN_R12_MATCHED_TABLE = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "figure_fcv_fcs_fu_sc_corr_forest_fcs_calibrated_skeleton_kmeans_nearest_r12_two_way_subsampling_input.csv"
)
OUTPUT_PNG = PACK_ROOT / "figures" / "figure_supplement_te_structural_controls.png"
STATS_CSV = (
    PACK_ROOT / "statistics" / "figure_supplement_te_structural_controls_stats.csv"
)

DIVISION_ORDER = list(ANATOMY_GROUP_ORDER)
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS.copy()
STATS_ROWS: list[dict] = []


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / sd


def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regions = pd.read_csv(REGIONS_FILE).sort_values("legacy_order")
    if len(regions) != 42 or regions["node"].duplicated().any():
        raise RuntimeError("Expected 42 unique canonical regions")
    if "rOB" in set(regions["node"]):
        raise RuntimeError("Canonical region set must exclude rOB")

    keys = ["root_area_id", "node", "anatomy_group"]
    functional = pd.read_csv(FUNCTIONAL_TABLE).merge(
        regions[keys], on=keys, how="inner", validate="many_to_one"
    )
    structural = pd.read_csv(STRUCTURAL_TABLE).merge(
        regions[keys], on=keys, how="inner", validate="many_to_one"
    )
    if len(functional) != 42 * 7 or len(structural) != 42 * 7:
        raise RuntimeError("Expected seven observations for each of 42 regions")

    functional_columns = ["EdgeStdFCV", "NetTE", "NeighborNetTE"]
    structural_columns = ["Reciprocity", "LogOutIn"]
    if functional[functional_columns].isna().any().any():
        raise RuntimeError("Functional control inputs contain missing values")
    if structural[structural_columns].dropna(how="all").empty:
        raise RuntimeError("Structural control inputs contain no finite values")

    functional["EdgeStdFCV_z"] = functional.groupby("recording_id")[
        "EdgeStdFCV"
    ].transform(lambda values: zscore(values.to_numpy(float)))
    functional["NetTE_z"] = functional.groupby("recording_id")["NetTE"].transform(
        lambda values: zscore(values.to_numpy(float))
    )
    functional["NeighborNetTE_z"] = functional.groupby("recording_id")[
        "NeighborNetTE"
    ].transform(lambda values: zscore(values.to_numpy(float)))
    structural["Reciprocity_z"] = structural.groupby("recording_id")["Reciprocity"].transform(
        lambda values: zscore(values.to_numpy(float))
    )
    structural["LogOutIn_z"] = structural.groupby("recording_id")["LogOutIn"].transform(
        lambda values: zscore(values.to_numpy(float))
    )

    functional_summary = (
        functional.groupby(keys, as_index=False)
        .agg(
            EdgeStdFCV=("EdgeStdFCV_z", "mean"),
            NetTE=("NetTE_z", "mean"),
            NeighborNetTE=("NeighborNetTE_z", "mean"),
        )
    )
    structural_summary = (
        structural.groupby(keys, as_index=False)
        .agg(
            Reciprocity=("Reciprocity", "mean"),
            LogOutIn=("LogOutIn", "mean"),
        )
    )
    region_summary = functional_summary.merge(
        structural_summary, on=keys, how="inner", validate="one_to_one"
    )
    region_summary["FCV_z"] = zscore(region_summary["EdgeStdFCV"].to_numpy(float))
    main_r12 = pd.read_csv(MAIN_R12_MATCHED_TABLE)[
        [*keys, "EdgeStdFCV", "Reciprocity", "LogOutIn"]
    ].rename(
        columns={
            "EdgeStdFCV": "Main_FCV",
            "Reciprocity": "Main_Reciprocity",
            "LogOutIn": "Main_LogOutIn",
        }
    )
    region_summary = region_summary.merge(
        main_r12, on=keys, how="inner", validate="one_to_one"
    )
    if len(region_summary) != 42:
        raise RuntimeError("Expected 42 regions after matching the main skeleton-r12 input")
    return functional, structural, region_summary


def wide_division_df(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = {
        division: df.loc[df["anatomy_group"].eq(division), value_col]
        .dropna()
        .to_numpy(float)
        for division in DIVISION_ORDER
    }
    max_len = max(len(values) for values in grouped.values())
    return pd.DataFrame(
        {
            division: pd.Series(values, dtype=float).reindex(range(max_len))
            for division, values in grouped.items()
        }
    )


def record_division_stats(panel: str, df: pd.DataFrame) -> tuple[list, np.ndarray, np.ndarray]:
    groups = [df[group].dropna().to_numpy(float) for group in DIVISION_ORDER]
    pairs = list(itertools.combinations(range(len(DIVISION_ORDER)), 2))
    pvals = [
        mannwhitneyu(groups[i], groups[j], alternative="two-sided").pvalue
        for i, j in pairs
    ]
    reject, corrected, _, _ = multipletests(pvals, method="holm")
    for index, (i, j) in enumerate(pairs):
        STATS_ROWS.append(
            {
                "figure": "figure_supplement_te_structural_controls",
                "SC_source": SC_SOURCE if panel in {"E: Reciprocity", "F: LogOutIn"} else "not_applicable",
                "panel": panel,
                "test": "Mann-Whitney U",
                "metric": panel,
                "group_1": DIVISION_ORDER[i],
                "group_2": DIVISION_ORDER[j],
                "n_group_1": len(groups[i]),
                "n_group_2": len(groups[j]),
                "p_uncorrected": pvals[index],
                "p_corrected": corrected[index],
                "correction": "Holm",
                "significant_0.05": bool(reject[index]),
            }
        )
    return pairs, reject, corrected


def add_significance_bars(ax, df: pd.DataFrame, panel: str) -> None:
    pairs, reject, corrected = record_division_stats(panel, df)
    significant = sorted(
        [(pairs[k], corrected[k]) for k in range(len(pairs)) if reject[k]],
        key=lambda item: item[0][1] - item[0][0],
    )
    if not significant:
        return
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for level, ((i, j), p_value) in enumerate(significant[:6]):
        y = y_max + y_range * (0.025 + level * 0.070)
        bar_height = y_range * 0.016
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.plot(
            [i, i, j, j], [y, y + bar_height, y + bar_height, y],
            lw=0.6, color="#333333", clip_on=False,
        )
        ax.text(
            (i + j) / 2, y + bar_height, stars, ha="center", va="center",
            fontsize=fs.SUPP_STAR_FS, clip_on=False,
        )
    ax.set_ylim(y_min, y_max)


def draw_boxplot(ax, df: pd.DataFrame, ylabel: str, panel: str, ylim: tuple) -> None:
    values, labels = [], []
    for division in DIVISION_ORDER:
        group_values = df[division].dropna().to_numpy(float)
        values.extend(group_values)
        labels.extend([division] * len(group_values))
    fs.draw_main_box_strip(
        ax,
        labels,
        values,
        DIVISION_ORDER,
        palette=[DIVISION_COLORS[group] for group in DIVISION_ORDER],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xticks(range(len(DIVISION_ORDER)))
    ax.set_xticklabels(DIVISION_ORDER)
    ax.set_ylim(*ylim)
    add_significance_bars(ax, df, panel)


def correlation_rows(region_summary: pd.DataFrame) -> list[dict]:
    specifications = [
        ("C", "TE_net", "NetTE", r"$\mathrm{TE}_{\mathrm{net}}$", "functional"),
        (
            "D", "Neighbor TE_net", "NeighborNetTE",
            "Neighbor " + r"$\mathrm{TE}_{\mathrm{net}}$", "functional",
        ),
        ("G", "Reciprocity", "Reciprocity", "Reciprocity", "structural"),
        (
            "H", "LogOutIn", "LogOutIn",
            r"$\log$(Out/In)", "structural",
        ),
    ]
    rows = []
    for panel, metric, column, label, family in specifications:
        if family == "structural":
            x_column = f"Main_{column}"
            y_column = "Main_FCV"
        else:
            x_column = column
            y_column = "EdgeStdFCV"
        sub = region_summary[["anatomy_group", y_column, x_column]].dropna()
        result = pearsonr(sub[y_column], sub[x_column])
        rows.append(
            {
                "figure": "figure_supplement_te_structural_controls",
                "SC_source": SC_SOURCE if family == "structural" else "not_applicable",
                "panel": panel,
                "test": "Pearson correlation",
                "metric": metric,
                "plot_label": label,
                "family": family,
                "x_values": sub[x_column].to_numpy(float),
                "y_values": (
                    sub[y_column].to_numpy(float)
                    if family == "structural"
                    else zscore(sub[y_column].to_numpy(float))
                ),
                "division_values": sub["anatomy_group"].astype(str).to_numpy(),
                "r": float(result.statistic),
                "p_value": float(result.pvalue),
                "n_regions": len(sub),
            }
        )

    for family, label in [
        ("functional", "functional directed-drive summary"),
        ("structural", "structural comparison summary"),
    ]:
        selected = [row for row in rows if row["family"] == family]
        reject, corrected, _, _ = multipletests(
            [row["p_value"] for row in selected], method="fdr_bh"
        )
        for row, is_significant, q_value in zip(selected, reject, corrected):
            row["p_corrected"] = float(q_value)
            row["correction"] = f"BH-FDR within {label}"
            row["significant_0.05"] = bool(is_significant)
    return rows


def draw_scatter(ax, row: dict, xlabel: str, show_legend: bool = False) -> None:
    x = zscore(row["x_values"])
    y = np.asarray(row["y_values"], dtype=float)
    divisions = np.asarray(row["division_values"], dtype=str)
    forest.scatter_by_division(ax, x, y, divisions)
    ax.text(
        0.62,
        0.20,
        f"r = {row['r']:.3f}\np {forest._format_p(row['p_value'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=forest.ANNOT_FS,
    )
    ax.set_xlabel(xlabel, fontsize=forest.AXIS_FS - 1, labelpad=2)
    ax.set_ylabel("FCV", fontsize=forest.AXIS_FS - 1, labelpad=2)
    ax.tick_params(axis="both", labelsize=forest.TICK_FS - 1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(
            frameon=True,
            edgecolor="black",
            fancybox=False,
            loc="upper left",
            ncol=1,
            fontsize=forest.ANNOT_FS - 1,
            handletextpad=0.3,
            labelspacing=0.25,
        )


def save_statistics(correlation_stats: list[dict]) -> None:
    rows = list(STATS_ROWS)
    excluded = {"plot_label", "family", "x_values", "y_values", "division_values"}
    rows.extend({key: value for key, value in row.items() if key not in excluded} for row in correlation_stats)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(STATS_CSV, index=False)


def main() -> None:
    fs.apply_supplement_figure_style()
    functional, structural, region_summary = load_clean_data()
    correlations = correlation_rows(region_summary)
    corr_by_panel = {row["panel"]: row for row in correlations}

    fig = plt.figure(figsize=(16.0, 7.0))
    grid = GridSpec(
        2, 4, figure=fig, left=0.075, right=0.975, top=0.91, bottom=0.14,
        hspace=0.52, wspace=0.42,
    )
    axes = {
        label: fig.add_subplot(grid[row, column])
        for label, row, column in [
            ("A", 0, 0), ("B", 0, 1), ("C", 0, 2), ("D", 0, 3),
            ("E", 1, 0), ("F", 1, 1), ("G", 1, 2), ("H", 1, 3),
        ]
    }

    draw_boxplot(
        axes["A"], wide_division_df(functional, "NetTE_z"),
        r"$\mathrm{TE}_{\mathrm{net}}$ (z)", "A: TE_net", (-3.4, 3.8),
    )
    draw_boxplot(
        axes["B"], wide_division_df(functional, "NeighborNetTE_z"),
        "Neighbor " + r"$\mathrm{TE}_{\mathrm{net}}$ (z)",
        "B: Neighbor TE_net", (-3.4, 3.8),
    )
    draw_scatter(axes["C"], corr_by_panel["C"], r"$\mathrm{TE}_{\mathrm{net}}$ (z)", True)
    draw_scatter(
        axes["D"], corr_by_panel["D"], "Neighbor " + r"$\mathrm{TE}_{\mathrm{net}}$ (z)"
    )
    draw_boxplot(
        axes["E"], wide_division_df(structural, "Reciprocity_z"),
        "Reciprocity", "E: Reciprocity", (-3.4, 3.8),
    )
    draw_boxplot(
        axes["F"], wide_division_df(structural, "LogOutIn_z"),
        r"$\log$(Out/In)", "F: LogOutIn", (-3.4, 3.8),
    )
    draw_scatter(axes["G"], corr_by_panel["G"], "Reciprocity")
    draw_scatter(axes["H"], corr_by_panel["H"], r"$\log$(Out/In)")

    for label, ax in axes.items():
        ax.text(
            -0.24, 1.08, label, transform=ax.transAxes,
            fontsize=fs.SUPP_PANEL_LABEL_FS, fontweight="bold", va="bottom", ha="left",
        )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", transparent=False)
    save_statistics(correlations)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {STATS_CSV}")


if __name__ == "__main__":
    main()
