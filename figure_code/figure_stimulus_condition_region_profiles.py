"""Draw stimulus condition profiles in the Figure Supply 10A bar style."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_style as fs
from figure_stimulus_condition_region_heatmaps import (
    CONDITION_ORDER,
    DIVISION_ORDER,
    OUT_TABLE,
    build_table,
)


PACK = Path(__file__).resolve().parents[1]
OUT_PNG = PACK / "figures" / "figure_stimulus_condition_region_profiles.png"
OUT_STATS = (
    PACK / "statistics" / "figure_stimulus_condition_region_profiles_summary.csv"
)
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS
PANEL_FS = fs.SUPP_PANEL_LABEL_FS
AXIS_FS = fs.SUPP_AXIS_FS
TICK_FS = fs.SUPP_TICK_FS

PANELS = [
    ("A", "FCV", "OMR forward", "FCV_raw_mean", "FCV_raw_sem"),
    ("B", "FCV", "OMR rightward", "FCV_raw_mean", "FCV_raw_sem"),
    ("C", "FCV", "OMR leftward", "FCV_raw_mean", "FCV_raw_sem"),
    ("D", "FCS", "OMR forward", "FCS_raw_mean", "FCS_raw_sem"),
    ("E", "FCS", "OMR rightward", "FCS_raw_mean", "FCS_raw_sem"),
    ("F", "FCS", "OMR leftward", "FCS_raw_mean", "FCS_raw_sem"),
]


def panel_data(table: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = table.loc[table["stimulus_label"].astype(str).eq(condition)].copy()
    return sub.sort_values(["division_order", "legacy_order"]).reset_index(drop=True)


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    ylabel: str,
    condition: str,
    mean_col: str,
    sem_col: str,
) -> None:
    x = np.arange(len(data))
    divisions = data["anatomy_group"].tolist()
    colors = [DIVISION_COLORS[group] for group in divisions]
    ax.bar(
        x,
        data[mean_col],
        yerr=data[sem_col],
        width=0.62,
        color=colors,
        alpha=0.86,
        error_kw={
            "ecolor": "#2F3437",
            "elinewidth": 0.8,
            "capsize": 1.8,
            "capthick": 0.8,
        },
        zorder=2,
    )
    for position in range(1, len(divisions)):
        if divisions[position] != divisions[position - 1]:
            ax.axvline(position - 0.5, color="#B5B5B5", linewidth=0.8, zorder=0)
    ax.axhline(0, color="#B5B5B5", linewidth=0.6, zorder=0)
    ax.set_xlim(-1.1, len(data))
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_title(condition, fontsize=AXIS_FS, fontweight="bold", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(data["node"], rotation=60, ha="right", fontsize=TICK_FS)
    for label, division in zip(ax.get_xticklabels(), divisions, strict=True):
        label.set_color(DIVISION_COLORS[division])
        if division == "Tel":
            label.set_fontweight("bold")
            label.set_fontstyle("italic")
    ax.tick_params(
        axis="both",
        direction="out",
        bottom=True,
        left=True,
        length=3,
        width=0.8,
        labelsize=TICK_FS,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.0)


def main() -> None:
    fs.apply_supplement_figure_style()
    table = build_table()
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    fig, axes = plt.subplots(6, 1, figsize=(16.0, 18.0), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.98, bottom=0.065, hspace=0.64)
    stats_rows = []
    for ax, (panel, measure, condition, mean_col, sem_col) in zip(
        axes, PANELS, strict=True
    ):
        data = panel_data(table, condition)
        if len(data) != 42 or not data["n_subjects"].eq(7).all():
            raise RuntimeError(f"Incomplete panel {panel}: {len(data)} regions")
        draw_panel(ax, data, measure, condition, mean_col, sem_col)
        fs.add_panel_label_fig(fig, ax, panel, dx=-0.055, dy=0.008, fontsize=PANEL_FS)
        stats_rows.append(
            data.assign(panel=panel, measure=measure)[
                [
                    "panel", "measure", "stimulus_label", "root_area_id", "node",
                    "anatomy_group", "n_subjects", mean_col, sem_col,
                ]
            ].rename(columns={mean_col: "mean", sem_col: "sem"})
        )

    # Use one common range within FCV and one within FCS for condition comparison.
    for group in (axes[:3], axes[3:]):
        low = min(ax.get_ylim()[0] for ax in group)
        high = max(ax.get_ylim()[1] for ax in group)
        for ax in group:
            ax.set_ylim(low, high)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(stats_rows, ignore_index=True).to_csv(OUT_STATS, index=False)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
