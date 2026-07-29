"""Mean and OMR-condition variability of stimulus FCV using FU-SC measures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.multitest import multipletests

import figure_fcv_fcs_sc_corr_forest as forest


PACK_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PACK_ROOT / "derived_data" / "common"
REGIONS = INPUT_DIR / "legacy_stimulus_forest_42_regions_no_rOB.csv"
SPONTANEOUS = PACK_ROOT / "derived_data" / "figure9" / "figure9_region_summary.csv"
STIMULUS = PACK_ROOT / "derived_data" / "figure_stimulus" / "stimulus_fc_region_summary.csv"
STIMULUS_DETAIL = (
    PACK_ROOT / "derived_data" / "figure_stimulus" / "stimulus_fc_measures_subject_condition_region.csv"
)
DEFAULT_SC_SOURCE = "fcs_calibrated_endpoint"
FIGURE_BASENAME = "figure_stimulus_delta_fcv_acd_combined"
PANEL_AB_DISTRIBUTION = "region_bootstrap"
PANEL_AB_P_MODE = "fdr_bh"

FU_SC_MEASURES = [
    ("Hard_OO_fraction", "OO frac."),
    ("FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("FU_DCApre", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    ("Reciprocity", "Reciprocity"),
    ("LogOutIn", "log(O/I)"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw stimulus FCV forest/scatter panels with FU-SC measures")
    parser.add_argument("--sc-source", default=DEFAULT_SC_SOURCE)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix added to output files. Defaults to the SC source for non-canonical sources.",
    )
    return parser.parse_args()


ARGS = _parse_args()
_suffix = ARGS.output_suffix or ("" if ARGS.sc_source == DEFAULT_SC_SOURCE else ARGS.sc_source)
OUTPUT_SUFFIX = f"_{_suffix}" if _suffix else ""
FU_STRUCTURAL = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "functional_unit_region_measures"
    / ARGS.sc_source
    / "figure12_functional_unit_region_summary.csv"
)
STIMULUS_INPUT = INPUT_DIR / f"{FIGURE_BASENAME}{OUTPUT_SUFFIX}_input.csv"
OUT_PNG = PACK_ROOT / "figures" / f"{FIGURE_BASENAME}{OUTPUT_SUFFIX}.png"
OUT_STATS = PACK_ROOT / "statistics" / f"{FIGURE_BASENAME}{OUTPUT_SUFFIX}_stats.csv"


def compute_cells(frame: pd.DataFrame, func_specs, analysis: str):
    cells = []
    for y_column, y_name, _ in func_specs:
        y = frame[y_column].to_numpy(float)
        for position, (sc_column, _) in enumerate(forest.SC_MEASURES):
            result = forest.bootstrap_pearson(frame[sc_column].to_numpy(float), y)
            cells.append(
                {
                    "analysis": analysis,
                    "y_column": y_column,
                    "y_name": y_name,
                    "sc": sc_column,
                    "position": position,
                    **result,
                }
            )
    for y_column, _, _ in func_specs:
        group = [cell for cell in cells if cell["y_column"] == y_column]
        corrected = multipletests([cell["p"] for cell in group], method="fdr_bh")[1]
        for cell, p_fdr in zip(group, corrected):
            cell["p_fdr"] = float(p_fdr)
    return cells


def _build_base_frame() -> pd.DataFrame:
    """Match the canonical stimulus set to complete FU structural measures."""
    sc_columns = [column for column, _ in FU_SC_MEASURES]
    regions = pd.read_csv(REGIONS)[["legacy_order", "root_area_id", "node", "anatomy_group"]]
    spontaneous = pd.read_csv(SPONTANEOUS)[["root_area_id", "node", "EdgeStdFCV", "FCS"]].rename(
        columns={"EdgeStdFCV": "spont_FCV", "FCS": "spont_FCS"}
    )
    stimulus = pd.read_csv(STIMULUS)[["root_area_id", "node", "FCV", "FCS"]].rename(
        columns={"FCV": "stim_FCV", "FCS": "stim_FCS"}
    )
    condition_fcv = (
        pd.read_csv(STIMULUS_DETAIL)
        .groupby(["root_area_id", "node", "stimulus_index"], as_index=False)["FCV_z"]
        .mean()
        .groupby(["root_area_id", "node"], as_index=False)["FCV_z"]
        .agg(lambda values: float(np.std(values.to_numpy(float), ddof=0)))
        .rename(columns={"FCV_z": "stim_FCV_condition_sd"})
    )
    structural = pd.read_csv(FU_STRUCTURAL)[["root_area_id", "node", *sc_columns]]
    frame = (
        regions
        .merge(spontaneous, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(stimulus, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(condition_fcv, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(structural, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
    )
    required = [
        "spont_FCV", "spont_FCS", "stim_FCV", "stim_FCS", "stim_FCV_condition_sd", *sc_columns
    ]
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if len(frame) < 4:
        raise RuntimeError("Too few complete regions after FU-SC matching")
    return frame.sort_values("legacy_order").reset_index(drop=True)


def _zscore_structural(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column, _ in FU_SC_MEASURES:
        out[f"{column}_raw"] = out[column]
        out[column] = forest._zscore(out[column].to_numpy(float))
    return out


def build_stimulus_frame() -> pd.DataFrame:
    frame = _build_base_frame()
    frame["EdgeStdFCV"] = forest._zscore(frame["stim_FCV"].to_numpy(float))
    frame["FCS"] = forest._zscore(frame["stim_FCS"].to_numpy(float))
    frame["ConditionFCVSD"] = forest._zscore(frame["stim_FCV_condition_sd"].to_numpy(float))
    return _zscore_structural(frame)


def draw_forest(ax, cells, y_column: str, title: str, color: str, panel: str):
    xpos = np.arange(len(forest.SC_MEASURES))
    ax.axvline(0, color="#888888", lw=0.8, zorder=1)
    for boundary in xpos[:-1] + 0.5:
        ax.axhline(boundary, color="#cccccc", lw=0.8, linestyle=(0, (4, 4)), zorder=0)

    group = sorted(
        [cell for cell in cells if cell["y_column"] == y_column],
        key=lambda cell: cell["position"],
    )
    centers = [xpos[cell["position"]] for cell in group]
    parts = ax.violinplot(
        [cell["boot"] for cell in group],
        positions=centers,
        widths=0.62,
        showextrema=False,
        vert=False,
    )
    forest._style_violin(parts, color)
    ax.scatter(
        [cell["r"] for cell in group],
        centers,
        s=24,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.8,
        zorder=5,
    )
    for center, cell in zip(centers, group):
        p_for_star = cell["p"] if PANEL_AB_P_MODE == "raw" else cell["p_fdr"]
        star = forest._star(p_for_star)
        right = cell["r"] >= 0
        x_at = cell["boot"].max() + 0.02 if right else cell["boot"].min() - 0.02
        ax.text(
            x_at,
            center,
            star,
            va="center",
            ha="left" if right else "right",
            fontsize=forest.ANNOT_FS + (1 if star != "n.s." else -1),
            color="#222222" if star != "n.s." else "#999999",
            fontweight="bold" if star != "n.s." else "normal",
            zorder=6,
        )
    ax.set_yticks(xpos)
    ax.set_yticklabels([label for _, label in forest.SC_MEASURES], fontsize=forest.TICK_FS)
    ax.set_ylim(len(forest.SC_MEASURES) - 0.4, -0.6)
    ax.margins(x=0.18)
    if panel in {"A", "B"}:
        ax.set_xlim(-1.0, 1.0)
        
    ax.tick_params(axis="both", labelsize=forest.TICK_FS)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=forest.AXIS_FS, color=color, fontweight="bold", loc="left")
    if panel == "B":
        ax.set_xlabel("Pearson correlation", fontsize=forest.AXIS_FS)
    ax.text(
        -0.30,
        1.0,
        panel,
        transform=ax.transAxes,
        fontsize=forest.PANEL_FS,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def draw_scatter(
    ax,
    frame: pd.DataFrame,
    cells,
    y_column: str,
    y_label: str,
    sc_column: str,
    sc_label: str,
    color: str,
    panel: str,
    show_legend: bool,
):
    x = frame[sc_column].to_numpy(float)
    y = frame[y_column].to_numpy(float)
    forest.scatter_by_division(ax, x, y, frame["anatomy_group"].to_numpy())
    cell = next(
        cell for cell in cells if cell["y_column"] == y_column and cell["sc"] == sc_column
    )
    
    dx=0.6;
    dy=0.96;
    if panel == "C" or panel == "D":
          dx=0.1;
          dy=1.01;

    ax.text(
        dx,
        dy,
        f"r = {cell['r']:.3f}\np {forest._format_p(cell['p'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=forest.ANNOT_FS,
    )
    ax.set_xlabel(sc_label, fontsize=forest.AXIS_FS, labelpad=2)
    ax.set_ylabel(y_label, fontsize=forest.AXIS_FS, labelpad=2, color=color)
    ax.tick_params(axis="both", labelsize=forest.TICK_FS)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[["top", "right"]].set_visible(False)
    
    
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.text(
        -0.34,
        1.0,
        panel,
        transform=ax.transAxes,
        fontsize=forest.PANEL_FS,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def main() -> None:
    forest.SC_MEASURES = FU_SC_MEASURES
    forest.SCAT_COLS = FU_SC_MEASURES[:2]
    stimulus = build_stimulus_frame()
    stimulus.to_csv(STIMULUS_INPUT, index=False)

    stimulus_specs = [
        ("EdgeStdFCV", "FCV", "#5B8DB8"),
        ("ConditionFCVSD", "FCV SD across OMR conditions", "#4C9A8A"),
    ]
    stimulus_cells = compute_cells(stimulus, stimulus_specs, "stimulus")

    figure = plt.figure(
        figsize=(forest.fs.MAIN_FIGURE_WIDTH, forest.fs.MAIN_FIGURE_HEIGHT_SHORT)
    )
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 1.0],
        wspace=0.42,
        hspace=0.32,
    )
    axes = [[figure.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)]

    draw_forest(axes[0][0], stimulus_cells, "EdgeStdFCV", "Stimulus: corr. with FCV", "#5B8DB8", "A")
    draw_scatter(
        axes[0][1], stimulus, stimulus_cells, "EdgeStdFCV", "FCV",
        "Hard_OO_fraction", "OO frac.", "#5B8DB8", "C", False,
    )
    draw_scatter(
        axes[0][2], stimulus, stimulus_cells, "EdgeStdFCV", "FCV",
        "FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$", "#5B8DB8", "D", True,
    )

    draw_forest(
        axes[1][0], stimulus_cells, "ConditionFCVSD",
        "OMR FCV s.d.", "#4C9A8A", "B",
    )
    draw_scatter(
        axes[1][1], stimulus, stimulus_cells, "ConditionFCVSD", "OMR FCV s.d.",
        "Hard_OO_fraction", "OO frac.", "#4C9A8A", "E", False,
    )
    draw_scatter(
        axes[1][2], stimulus, stimulus_cells, "ConditionFCVSD", "OMR FCV s.d.",
        "FU_DCApost", r"$\mathrm{DCA}_{\mathrm{post}}$", "#4C9A8A", "F", False,
    )

    axes[0][1].set_ylim((-2.0,2.0));
    axes[0][2].set_ylim((-2.0,2.0));
    legend_handles, legend_labels = axes[0][1].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.0),
        ncol=4,
        fontsize=forest.TICK_FS,
        frameon=True,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=False)
    plt.close(figure)

    rows = []
    for cell in stimulus_cells:
        rows.append(
            {
                "analysis": cell["analysis"],
                "func": cell["y_name"],
                "sc": cell["sc"],
                "coef": cell["r"],
                "p": cell["p"],
                "p_fdr_bh_within_analysis": cell["p_fdr"],
                "panel_ab_p_used": cell["p"] if PANEL_AB_P_MODE == "raw" else cell["p_fdr"],
                "panel_ab_p_mode": PANEL_AB_P_MODE,
                "boot_ci_lo": cell["lo"],
                "boot_ci_hi": cell["hi"],
                "n": len(stimulus),
                "n_boot": forest.N_BOOT,
                "bootstrap_seed": forest.BOOTSTRAP_SEED,
                "panel_ab_distribution": PANEL_AB_DISTRIBUTION,
                "SC_source": ARGS.sc_source,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_STATS, index=False)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
