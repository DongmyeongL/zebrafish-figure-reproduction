#!/usr/bin/env python3
"""Draw regional zebrafish structural measures from clean Figure 12 data."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from shared.figure9_anatomy import ANATOMY_GROUP_ORDER

import figure_style as fs


STRUCTURAL_TABLE = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "figure12_subject_region_structural_measures.csv"
)
REGIONS_FILE = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
OUTPUT_PNG = PACK_ROOT / "figures" / "figure_supply_2_proc.png"
OUTPUT_STATS = PACK_ROOT / "statistics" / "figure_supply_2_proc_region_summary.csv"

FEATURES = (
    ("A", "PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("B", "PreDCA", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    ("C", "Modularity", "Modularity Q"),
    ("D", "LogOutIn", r"$\log$(Out/In)"),
    ("E", "OO_fraction", "OO fraction"),
)
DIVISION_RANK = {group: rank for rank, group in enumerate(ANATOMY_GROUP_ORDER)}
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS

AXIS_FS = fs.SUPP_AXIS_FS
TICK_FS = fs.SUPP_TICK_FS
PANEL_FS = fs.SUPP_PANEL_LABEL_FS


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the clean subject-level table and canonical regions."""
    structural = pd.read_csv(STRUCTURAL_TABLE)
    regions = pd.read_csv(REGIONS_FILE).sort_values("legacy_order").copy()

    required_structural = {"Subject", "root_area_id", "node", "anatomy_group"}
    required_structural.update(metric for _, metric, _ in FEATURES)
    missing = required_structural.difference(structural.columns)
    if missing:
        raise RuntimeError(f"Structural table is missing columns: {sorted(missing)}")
    if not {"legacy_order", "root_area_id", "node", "anatomy_group"}.issubset(regions.columns):
        raise RuntimeError("Canonical region table has an unexpected schema")
    if len(regions) != 42 or regions["node"].duplicated().any():
        raise RuntimeError("Expected 42 unique canonical regions")
    if "rOB" in set(regions["node"]):
        raise RuntimeError("Canonical region set must exclude rOB")

    data = structural.merge(
        regions[["legacy_order", "root_area_id", "node", "anatomy_group"]],
        on=["root_area_id", "node", "anatomy_group"],
        how="inner",
        validate="many_to_one",
    )
    counts = data.groupby("node")["Subject"].nunique()
    if len(counts) != len(regions) or not counts.eq(7).all():
        raise RuntimeError("Every canonical region must contain all seven subjects")

    regions["division_rank"] = regions["anatomy_group"].map(DIVISION_RANK)
    if regions["division_rank"].isna().any():
        unknown = regions.loc[regions["division_rank"].isna(), "anatomy_group"].unique()
        raise RuntimeError(f"Unknown anatomical divisions: {unknown.tolist()}")
    regions = regions.sort_values(["division_rank", "legacy_order"]).reset_index(drop=True)
    return data, regions


def summarize(data: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel, metric, ylabel in FEATURES:
        for region in regions.itertuples(index=False):
            values = pd.to_numeric(
                data.loc[data["node"].eq(region.node), metric], errors="coerce"
            ).to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "figure": "figure_supply_2_proc",
                    "panel": panel,
                    "metric": metric,
                    "metric_label": ylabel,
                    "root_area_id": int(region.root_area_id),
                    "region": region.node,
                    "division": region.anatomy_group,
                    "n_subjects": int(values.size),
                    "mean": float(np.mean(values)) if values.size else np.nan,
                    "sem": (
                        float(np.std(values, ddof=1) / np.sqrt(values.size))
                        if values.size > 1
                        else np.nan
                    ),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "median": float(np.median(values)) if values.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def draw_panel(ax, panel_data: pd.DataFrame, ylabel: str) -> None:
    x = np.arange(len(panel_data))
    colors = [DIVISION_COLORS[group] for group in panel_data["division"]]
    ax.bar(
        x,
        panel_data["mean"],
        yerr=panel_data["sem"],
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

    divisions = panel_data["division"].tolist()
    for position in range(1, len(divisions)):
        if divisions[position] != divisions[position - 1]:
            ax.axvline(position - 0.5, color="#B5B5B5", lw=0.8, zorder=0)

    ax.axhline(0, color="#B5B5B5", lw=0.6, zorder=0)
    ax.set_xlim(-1.1, len(panel_data))
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_xticks(x)
    ax.set_xticklabels(
        panel_data["region"], rotation=60, ha="right", fontsize=TICK_FS
    )
    for label, division in zip(ax.get_xticklabels(), divisions):
        label.set_color(DIVISION_COLORS[division])
        if division == "Tel":
            label.set_fontweight("bold")
            label.set_fontstyle("italic")
    ax.tick_params(
        axis="both", direction="out", bottom=True, left=True, length=3, width=0.8,
        labelsize=TICK_FS,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def main() -> None:
    fs.apply_supplement_figure_style()

    data, regions = load_data()
    summary = summarize(data, regions)

    fig, axes = plt.subplots(5, 1, figsize=(16.0, 15.0), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.975, bottom=0.075, hspace=0.52)
    for ax, (panel, _, ylabel) in zip(axes, FEATURES):
        panel_data = summary.loc[summary["panel"].eq(panel)].copy()
        draw_panel(ax, panel_data, ylabel)
        fs.add_panel_label_fig(fig, ax, panel, dx=-0.055, dy=0.008, fontsize=PANEL_FS)

    for ax in axes[:2]:
        ax.set_yticks([-0.04, -0.02, 0.00, 0.02])

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_STATS, index=False)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
