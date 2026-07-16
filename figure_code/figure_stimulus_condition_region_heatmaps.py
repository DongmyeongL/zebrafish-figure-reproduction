"""SI heatmaps of stimulus FCV/FCS across the canonical 42 regions."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import figure_style as fs


PACK = Path(__file__).resolve().parents[1]
MEASURES = (
    PACK
    / "derived_data"
    / "figure_stimulus"
    / "stimulus_fc_measures_subject_condition_region.csv"
)
REGIONS = (
    PACK
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
OUT_TABLE = (
    PACK
    / "derived_data"
    / "figure_stimulus"
    / "stimulus_condition_region_heatmap_input.csv"
)
OUT_PNG = PACK / "figures" / "figure_stimulus_condition_region_heatmaps.png"

CONDITION_ORDER = ["OMR forward", "OMR rightward", "OMR leftward"]
CONDITION_LABELS = ["Forward", "Rightward", "Leftward"]
DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd


def build_table() -> pd.DataFrame:
    measures = pd.read_csv(MEASURES).replace([np.inf, -np.inf], np.nan)
    regions = pd.read_csv(REGIONS)
    if len(regions) != 42 or regions["node"].eq("rOB").any():
        raise RuntimeError("Expected the canonical 42-region set without rOB.")

    matched = measures.merge(
        regions[["legacy_order", "root_area_id", "node", "anatomy_group"]],
        on=["root_area_id", "node"],
        how="inner",
        validate="many_to_one",
    )
    grouped = (
        matched.groupby(
            ["stimulus_label", "legacy_order", "root_area_id", "node", "anatomy_group"],
            as_index=False,
        )
        .agg(
            n_subjects=("subject", "nunique"),
            FCV_raw_mean=("FCV_raw", "mean"),
            FCV_raw_sem=(
                "FCV_raw", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
            ),
            FCS_raw_mean=("FCS_raw", "mean"),
            FCS_raw_sem=(
                "FCS_raw", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
            ),
            FCV_subject_z_mean=("FCV_z", "mean"),
            FCV_subject_z_sem=(
                "FCV_z", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
            ),
            FCS_subject_z_mean=("FCS_z", "mean"),
            FCS_subject_z_sem=(
                "FCS_z", lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum())
            ),
        )
    )
    grouped["FCV_heatmap_z"] = grouped.groupby("stimulus_label")[
        "FCV_subject_z_mean"
    ].transform(zscore)
    grouped["FCS_heatmap_z"] = grouped.groupby("stimulus_label")[
        "FCS_subject_z_mean"
    ].transform(zscore)
    grouped["stimulus_label"] = pd.Categorical(
        grouped["stimulus_label"], categories=CONDITION_ORDER, ordered=True
    )
    grouped["division_order"] = grouped["anatomy_group"].map(
        {group: idx for idx, group in enumerate(DIVISION_ORDER)}
    )
    grouped = grouped.sort_values(
        ["division_order", "legacy_order", "stimulus_label"]
    ).reset_index(drop=True)

    counts = grouped.groupby("stimulus_label", observed=True).size()
    if not counts.eq(42).all() or not grouped["n_subjects"].eq(7).all():
        raise RuntimeError(
            f"Incomplete condition-region table: rows={counts.to_dict()}, "
            f"subject range={grouped['n_subjects'].min()}-{grouped['n_subjects'].max()}"
        )
    return grouped


def matrix(table: pd.DataFrame, value: str, region_order: list[str]) -> np.ndarray:
    return (
        table.pivot(index="node", columns="stimulus_label", values=value)
        .reindex(index=region_order, columns=CONDITION_ORDER)
        .to_numpy(float)
    )


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    panel: str,
    region_order: list[str],
    divisions: list[str],
    norm: TwoSlopeNorm,
    show_ylabels: bool,
) -> None:
    image = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(3), CONDITION_LABELS, rotation=32, ha="right")
    ax.set_yticks(np.arange(len(region_order)))
    if show_ylabels:
        ax.set_yticklabels(region_order)
        for tick, division in zip(ax.get_yticklabels(), divisions, strict=True):
            tick.set_color(DIVISION_COLORS[division])
            tick.set_fontweight("bold" if division == "Tel" else "normal")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_title(title, fontsize=fs.MAIN_TITLE_FS, fontweight="bold", pad=8)
    ax.text(
        -0.22 if show_ylabels else -0.13,
        1.025,
        panel,
        transform=ax.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    boundaries = [
        idx - 0.5
        for idx in range(1, len(divisions))
        if divisions[idx] != divisions[idx - 1]
    ]
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def main() -> None:
    fs.apply_main_figure_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    table = build_table()
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    region_info = (
        table[["division_order", "legacy_order", "node", "anatomy_group"]]
        .drop_duplicates()
        .sort_values(["division_order", "legacy_order"])
    )
    region_order = region_info["node"].tolist()
    divisions = region_info["anatomy_group"].tolist()
    fcv = matrix(table, "FCV_heatmap_z", region_order)
    fcs = matrix(table, "FCS_heatmap_z", region_order)
    limit = float(np.nanmax(np.abs(np.concatenate([fcv.ravel(), fcs.ravel()]))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 8.3), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.95, top=0.94, bottom=0.13, wspace=0.42)
    im_a = draw_heatmap(
        axes[0], fcv, "Stimulus FCV", "A", region_order, divisions, norm, True
    )
    draw_heatmap(
        axes[1], fcs, "Stimulus FCS", "B", region_order, divisions, norm, False
    )
    colorbar = fig.colorbar(
        im_a,
        ax=axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.075,
        aspect=35,
    )
    colorbar.set_label("Regional z-score")
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.tick_params(length=2, width=0.6)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=False)
    plt.close(fig)
    print(f"Saved {OUT_TABLE}")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
