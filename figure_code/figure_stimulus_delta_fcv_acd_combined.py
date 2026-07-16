"""Combine panels A, C, and D from the stimulus and delta forest figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.multitest import multipletests

import figure_fcv_fcs_sc_corr_forest as forest


PACK_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PACK_ROOT / "derived_data" / "common"
STIMULUS_INPUT = (
    INPUT_DIR / "figure_stimulus_fcv_fcs_sc_corr_forest_legacy42_no_rOB_input.csv"
)
DELTA_INPUT = (
    INPUT_DIR / "figure_delta_fcv_fcs_sc_corr_forest_legacy42_no_rOB_input.csv"
)
OUT_PNG = PACK_ROOT / "figures" / "figure_stimulus_delta_fcv_acd_combined.png"
OUT_STATS = (
    PACK_ROOT / "statistics" / "figure_stimulus_delta_fcv_acd_combined_stats.csv"
)


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
    corrected = multipletests([cell["p"] for cell in cells], method="fdr_bh")[1]
    for cell, p_fdr in zip(cells, corrected):
        cell["p_fdr"] = float(p_fdr)
    return cells


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
        star = forest._star(cell["p_fdr"])
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
    if panel == "A":
        ax.set_xlim(-1.0, 1.0)
    elif panel == "B":
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
    stimulus = pd.read_csv(STIMULUS_INPUT)
    delta = pd.read_csv(DELTA_INPUT)
    if len(stimulus) != 42 or len(delta) != 42:
        raise RuntimeError("Both combined-figure inputs must contain 42 regions")

    stimulus_specs = [
        ("EdgeStdFCV", "FCV", "#5B8DB8"),
        ("FCS", "FCS", "#E0A03C"),
    ]
    delta_specs = [
        ("EdgeStdFCV", r"$\Delta$FCV", "#5B8DB8"),
        ("FCS", r"$\Delta$FCS", "#E0A03C"),
    ]
    stimulus_cells = compute_cells(stimulus, stimulus_specs, "stimulus")
    delta_cells = compute_cells(delta, delta_specs, "delta")

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
        "OO_fraction", "OO frac.", "#5B8DB8", "C", False,
    )
    draw_scatter(
        axes[0][2], stimulus, stimulus_cells, "EdgeStdFCV", "FCV",
        "PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$", "#5B8DB8", "D", True,
    )

    draw_forest(axes[1][0], delta_cells, "EdgeStdFCV", r"Change: corr. with $\Delta$FCV", "#5B8DB8", "B")
    draw_scatter(
        axes[1][1], delta, delta_cells, "EdgeStdFCV", r"$\Delta$FCV",
        "OO_fraction", "OO frac.", "#5B8DB8", "E", False,
    )
    draw_scatter(
        axes[1][2], delta, delta_cells, "EdgeStdFCV", r"$\Delta$FCV",
        "PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$", "#5B8DB8", "F", False,
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
    figure.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(figure)

    rows = []
    for cell in [*stimulus_cells, *delta_cells]:
        rows.append(
            {
                "analysis": cell["analysis"],
                "func": cell["y_name"],
                "sc": cell["sc"],
                "coef": cell["r"],
                "p": cell["p"],
                "p_fdr_bh_within_analysis": cell["p_fdr"],
                "boot_ci_lo": cell["lo"],
                "boot_ci_hi": cell["hi"],
                "n": 42,
                "n_boot": forest.N_BOOT,
                "bootstrap_seed": forest.BOOTSTRAP_SEED,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_STATS, index=False)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
