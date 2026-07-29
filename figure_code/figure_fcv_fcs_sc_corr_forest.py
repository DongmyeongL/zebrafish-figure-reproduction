"""FCV/FCS correlations with five SC measures in 42 regions excluding rOB."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import figure_style as fs


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import ANATOMY_GROUP_ORDER


COMMON_REGIONS = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
FC_TABLE = PACK_ROOT / "derived_data" / "figure9" / "figure9_region_summary.csv"
SC_TABLE = PACK_ROOT / "derived_data" / "figure12" / "figure12_region_summary.csv"
MATCHED_TABLE = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "figure_fcv_fcs_sc_corr_forest_42regions_no_rOB_input.csv"
)
OUT_PNG = PACK_ROOT / "figures" / "figure_fcv_fcs_sc_corr_forest.png"
OUT_CSV = PACK_ROOT / "statistics" / "figure_fcv_fcs_sc_corr_forest_stats.csv"

SC_MEASURES = [
    ("OO_fraction", "OO frac."),
    ("PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("PreDCA", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    ("Modularity", "Mod Q"),
    ("LogOutIn", "log(O/I)"),
]
FUNC = [("EdgeStdFCV", "FCV", "#5B8DB8"), ("FCS", "FCS", "#E0A03C")]
SCAT_COLS = [("OO_fraction", "OO frac."), ("PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$")]
DIVISION_ORDER = list(ANATOMY_GROUP_ORDER)
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS.copy()
DIVISION_MARKERS = {"Tel": "o", "Di": "s", "Mes": "^", "Hind": "D"}
N_BOOT = 2000
BOOTSTRAP_SEED = 0
PANEL_AB_P_MODE = "fdr_bh"

ANNOT_FS = fs.TICK_FS_2COL - 1
TICK_FS = fs.TICK_FS_2COL
AXIS_FS = fs.AXIS_LABEL_FS_2COL
PANEL_FS = fs.PANEL_LABEL_FS_2COL

fs.apply_main_figure_style()
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.cal": "Nimbus Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _zscore(values):
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        raise ValueError("Cannot z-score a constant or non-finite measure")
    return (values - np.nanmean(values)) / sd


def _format_p(p):
    return f"{p:.3g}" if p >= 0.001 else "< 0.001"


def _star(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _style_violin(parts, color):
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.5)
        body.set_alpha(0.85)


def bootstrap_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 4:
        raise ValueError(f"Pearson bootstrap requires >=4 matched values, received {len(x)}")
    r_full, p_full = pearsonr(x, y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.full(N_BOOT, np.nan, dtype=float)
    for index in range(N_BOOT):
        sample = rng.integers(0, len(x), len(x))
        xb, yb = x[sample], y[sample]
        if np.std(xb) > 0 and np.std(yb) > 0:
            boot[index] = pearsonr(xb, yb).statistic
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "r": float(r_full),
        "p": float(p_full),
        "boot": boot,
        "lo": float(lo),
        "hi": float(hi),
    }


def load_frame():
    common = pd.read_csv(COMMON_REGIONS)
    fc = pd.read_csv(FC_TABLE)[["root_area_id", "node", "EdgeStdFCV", "FCS"]]
    sc_columns = [column for column, _ in SC_MEASURES]
    sc = pd.read_csv(SC_TABLE)[["root_area_id", "node"] + sc_columns]
    frame = (
        common[["root_area_id", "node", "anatomy_group"]]
        .merge(fc, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(sc, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
    )
    measures = [column for column, _, _ in FUNC] + sc_columns
    if len(frame) != 42 or frame["node"].nunique() != 42:
        raise RuntimeError(f"Common-region merge produced {len(frame)} rows instead of 42")
    if (frame["node"] == "rOB").any():
        raise RuntimeError("The canonical 42-region set must exclude rOB")
    if frame[measures].isna().any().any():
        missing = frame.loc[frame[measures].isna().any(axis=1), ["node"] + measures]
        raise RuntimeError(f"Missing values in complete-region forest input:\n{missing}")

    for column in measures:
        frame[f"{column}_raw"] = frame[column]
        frame[column] = _zscore(frame[column].to_numpy(float))
    frame = frame.sort_values("root_area_id").reset_index(drop=True)
    MATCHED_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MATCHED_TABLE, index=False)
    return frame


def scatter_by_division(ax, x, y, divisions):
    for division in DIVISION_ORDER:
        mask = divisions == division
        ax.scatter(
            x[mask],
            y[mask],
            s=64 if division == "Tel" else 42,
            color=DIVISION_COLORS[division],
            marker=DIVISION_MARKERS[division],
            alpha=0.90 if division == "Tel" else 0.78,
            edgecolor="black",
            linewidth=0.4,
            label=division,
            zorder=3,
        )
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min() - 0.15, x.max() + 0.15, 100)
    ax.plot(xx, slope * xx + intercept, color="#4d4d4d", lw=1.8, zorder=4)
    ax.axhline(0, color="#dddddd", lw=1, zorder=0)
    ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.margins(x=0.12, y=0.14)


def make_figure(frame, stats_rows):
    xpos = np.arange(len(SC_MEASURES))
    cells = []
    for y_column, y_name, _ in FUNC:
        y = frame[y_column].to_numpy(float)
        for position, (sc_column, _) in enumerate(SC_MEASURES):
            result = bootstrap_pearson(frame[sc_column].to_numpy(float), y)
            cells.append(
                {
                    "y_column": y_column,
                    "y_name": y_name,
                    "sc": sc_column,
                    "position": position,
                    **result,
                }
            )
    for y_column, _, _ in FUNC:
        group = [cell for cell in cells if cell["y_column"] == y_column]
        corrected = multipletests([cell["p"] for cell in group], method="fdr_bh")[1]
        for cell, p_fdr in zip(group, corrected):
            cell["p_fdr"] = float(p_fdr)
    for cell in cells:
        stats_rows.append(
            {
                "figure": "fcv_fcs_sc_corr_forest",
                "region_subset": "legacy_stimulus_forest_42regions_no_rOB",
                "func": cell["y_name"],
                "sc": cell["sc"],
                "method": "pearson",
                "coef": cell["r"],
                "p": cell["p"],
                "p_fdr_bh": cell["p_fdr"],
                "panel_ab_p_used": cell["p"] if PANEL_AB_P_MODE == "raw" else cell["p_fdr"],
                "panel_ab_p_mode": PANEL_AB_P_MODE,
                "boot_ci_lo": cell["lo"],
                "boot_ci_hi": cell["hi"],
                "n": len(frame),
                "n_boot": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )

    lookup = {(cell["y_column"], cell["sc"]): cell for cell in cells}
    figure = plt.figure(figsize=(fs.MAIN_FIGURE_WIDTH, fs.MAIN_FIGURE_HEIGHT_SHORT))
    grid = figure.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], wspace=0.42, hspace=0.32)

    for row, ((y_column, y_name, color), panel) in enumerate(zip(FUNC, "AB")):
        ax = figure.add_subplot(grid[row, 0])
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
        _style_violin(parts, color)
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
            star = _star(p_for_star)
            right = cell["r"] >= 0
            x_at = cell["boot"].max() + 0.02 if right else cell["boot"].min() - 0.02
            ax.text(
                x_at,
                center,
                star,
                va="center",
                ha="left" if right else "right",
                fontsize=ANNOT_FS + (1 if star != "n.s." else -1),
                color="#222222" if star != "n.s." else "#999999",
                fontweight="bold" if star != "n.s." else "normal",
                zorder=6,
            )
        ax.set_yticks(xpos)
        ax.set_yticklabels([label for _, label in SC_MEASURES], fontsize=TICK_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_ylim(len(SC_MEASURES) - 0.4, -0.6)
        ax.margins(x=0.16)
        ax.set_xlim((-1.0, 1.0));
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"corr. with {y_name}", fontsize=AXIS_FS, color=color, fontweight="bold", loc="left")
        ax.text(-0.30, 1.0, panel, transform=ax.transAxes, fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="right")
        if row == 1:
            ax.set_xlabel("Pearson correlation", fontsize=AXIS_FS)

    divisions = frame["anatomy_group"].to_numpy()
    panel_letters = iter("CDEF")
    legend_handles = None
    legend_labels = None
    for row, (y_column, y_name, color) in enumerate(FUNC):
        y = frame[y_column].to_numpy(float)
        for column, (sc_column, sc_label) in enumerate(SCAT_COLS):
            ax = figure.add_subplot(grid[row, column + 1])
            x = frame[sc_column].to_numpy(float)
            scatter_by_division(ax, x, y, divisions)
            cell = lookup[(y_column, sc_column)]
            
            pos_x=0.04;
            pos_y=0.96;
            if row == 0 and column == 0:
                pos_x = 0.66
                pos_y = 0.2
            
            if row == 1 and column == 0:
                pos_x = 0.20
                pos_y = 0.17

            if row == 1 and column == 1:
                pos_x = 0.05
                pos_y = 0.45
                                
                
            ax.text(
                pos_x,
                pos_y,
                f"r = {cell['r']:.3f}\np {_format_p(cell['p'])}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOT_FS,
            )
            ax.set_xlabel(sc_label, fontsize=AXIS_FS, labelpad=2)
            ax.set_ylabel(y_name, fontsize=AXIS_FS, labelpad=2, color=color)
            ax.tick_params(axis="both", labelsize=TICK_FS)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.spines[["top", "right"]].set_visible(False)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.text(
                -0.34,
                1.0,
                next(panel_letters),
                transform=ax.transAxes,
                fontsize=PANEL_FS,
                fontweight="bold",
                va="bottom",
                ha="right",
            )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.01),
        ncol=4,
        fontsize=TICK_FS,
        frameon=True,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    return figure


def main():
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    stats_rows = []
    figure = make_figure(frame, stats_rows)
    figure.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=False)
    plt.close(figure)
    pd.DataFrame(stats_rows).to_csv(OUT_CSV, index=False)
    print(f"n = {len(frame)} complete regions")
    print(f"Saved {MATCHED_TABLE}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
