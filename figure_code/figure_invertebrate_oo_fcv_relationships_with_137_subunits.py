"""Combine the invertebrate main panels with the 137-subunit fly analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figure_invertebrate_oo_fcv_relationships as base


ROOT = Path(__file__).resolve().parents[1]
SUBUNIT_DATA = ROOT / "derived_data" / "invertebrates" / "drosophila_ito_137_subunit_region_metrics.csv"
OUT = ROOT / "figures" / "figure_invertebrate_oo_fcv_relationships_with_137_subunits.png"
OUT_STATS = ROOT / "statistics" / "invertebrates" / "figure_invertebrate_oo_fcv_relationships_with_137_subunits_stats.csv"


def add_panel_labels(fig, axes):
    for label, ax in axes.items():
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.06,
            pos.y1 - 0.004,
            label,
            fontsize=base.PANEL_FS,
            fontweight="bold",
            ha="right",
            va="bottom",
        )


def add_row_title(fig, row_axes, text, y_offset=0.105):
    x0 = min(ax.get_position().x0 for ax in row_axes)
    x1 = max(ax.get_position().x1 for ax in row_axes)
    y1 = max(ax.get_position().y1 for ax in row_axes)
    fig.text(
        (x0 + x1) / 2,
        y1 + y_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=base.AXIS_FS,
        fontweight="bold",
    )


def tag(row, scale):
    if row is not None:
        row["analysis_scale"] = scale
    return row


def main():
    data = base._load_data()
    subunit = pd.read_csv(SUBUNIT_DATA).replace([np.inf, -np.inf], np.nan)
    subunit["species"] = "Drosophila"
    subunit["oo_level"] = "137-subunit SC summarized by side-aware region"

    fig = plt.figure(figsize=(8, 11.4))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.05, 1.05, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.085,
        right=0.985,
        top=0.885,
        bottom=0.055,
        wspace=0.34,
        hspace=0.96,
    )
    axes = {
        label: fig.add_subplot(gs[row, col])
        for row, labels in enumerate(["ABC", "DEF", "GHI"])
        for col, label in enumerate(labels)
    }

    rows = []
    rows.append(tag(base._draw_scatter(axes["A"], data, "C. elegans", "OO_fraction", "EdgeStdFCV", "OO fraction", "FCV"), "fine_class"))
    rows.append(tag(base._draw_scatter(axes["B"], data, "C. elegans", "PostDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{post}}$", "FCV"), "fine_class"))
    rows.append(tag(base._draw_scatter(axes["C"], data, "C. elegans", "PreDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{pre}}$", "FCV", text_offset=[0.60, 0.95]), "fine_class"))

    rows.append(tag(base._draw_scatter(axes["D"], data, "Drosophila", "OO_fraction", "EdgeStdFCV", "OO fraction", "FCV"), "local_cell"))
    rows.append(tag(base._draw_scatter(axes["E"], data, "Drosophila", "PostDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{post}}$", "FCV", text_offset=[0.20, 0.20]), "local_cell"))
    rows.append(tag(base._draw_scatter(axes["F"], data, "Drosophila", "PreDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{pre}}$", "FCV", text_offset=[0.50, 0.20]), "local_cell"))

    rows.append(tag(base._draw_scatter(axes["G"], subunit, "Drosophila", "OO_fraction_subunit", "EdgeStdFCV", "OO fraction", "FCV", text_offset=[0.45, 0.95]), "subunit_137"))
    rows.append(tag(base._draw_scatter(axes["H"], subunit, "Drosophila", "PostDCA_subunit", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{post}}$", "FCV", text_offset=[0.05, 0.95]), "subunit_137"))
    rows.append(tag(base._draw_scatter(axes["I"], subunit, "Drosophila", "PreDCA_subunit", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{pre}}$", "FCV", text_offset=[0.05, 0.95]), "subunit_137"))

    axes["A"].set_xlim(0.2, 0.8)
    axes["D"].set_xlim(0.0, 0.6)
    axes["G"].set_xlim(-0.05, 0.65)

    fig.canvas.draw()
    for label in "GHI":
        pos = axes[label].get_position()
        axes[label].set_position([pos.x0, pos.y0 + 0.065, pos.width, pos.height])
    fig.canvas.draw()
    top_axes = [axes[x] for x in "ABC"]
    middle_axes = [axes[x] for x in "DEF"]
    bottom_axes = [axes[x] for x in "GHI"]
    add_row_title(fig, top_axes, r"$\it{C.\ elegans}$")
    add_row_title(fig, middle_axes, r"$\it{Drosophila}$ : Local  SC base", y_offset=0.020)
    add_row_title(fig, bottom_axes, r"$\it{Drosophila}$ :  Region RC base", y_offset=0.020)

    base._add_row_legend(fig, top_axes, "C. elegans", ncol=4, y_offset=0.045)
    base._add_row_legend(fig, middle_axes, "Drosophila", ncol=3, y_offset=0.045)
    add_panel_labels(fig, axes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white", transparent=False)
    pd.DataFrame([row for row in rows if row is not None]).to_csv(OUT_STATS, index=False)
    plt.close(fig)
    print(f"Saved {OUT}")
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
