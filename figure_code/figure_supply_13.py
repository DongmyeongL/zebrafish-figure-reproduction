#!/usr/bin/env python3
"""Plot whole-network statistical validation of zebrafish NetTE."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

import figure_style as fs
from shared.figure9_anatomy import ANATOMY_GROUP_ORDER, anatomy_group


DATA_DIR = PACK_ROOT / "derived_data" / "figure_supply_13"
REGION_FILE = (
    PACK_ROOT / "derived_data" / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
OUT_PNG = PACK_ROOT / "figures" / "figure_supply_13.png"
DIVISION_COLORS = fs.ZEBRAFISH_DIVISION_COLORS
DIVISION_ORDER = list(ANATOMY_GROUP_ORDER)


def panel_label(ax, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=fs.SUPP_PANEL_LABEL_FS,
            fontweight="bold", va="top", ha="left")


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fs.SUPP_TICK_FS)


def draw_matrix(ax, matrix_data) -> None:
    matrix = matrix_data["net_te_matrix"].astype(float)
    nodes = matrix_data["unit_nodes"].astype(str)
    groups = np.array([anatomy_group(node) for node in nodes])
    order_parts = []
    for division in DIVISION_ORDER:
        indices = np.flatnonzero(groups == division)
        indices = indices[np.argsort(nodes[indices], kind="stable")]
        order_parts.append(indices)
    order = np.concatenate(order_parts)
    sorted_matrix = matrix[np.ix_(order, order)]
    limit = np.nanquantile(np.abs(sorted_matrix), 0.995)
    image = ax.imshow(sorted_matrix, cmap="RdBu_r", vmin=-limit, vmax=limit,
                      interpolation="nearest", aspect="auto")
    sizes = [len(indices) for indices in order_parts]
    boundaries = np.cumsum(sizes)[:-1] - 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="black", lw=0.6)
        ax.axvline(boundary, color="black", lw=0.6)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    centers = starts + np.asarray(sizes) / 2 - 0.5
    ax.set_xticks(centers, DIVISION_ORDER)
    ax.set_yticks(centers, DIVISION_ORDER)
    ax.set_xlabel("Target functional unit")
    ax.set_ylabel("Source functional unit")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("NetTE (bits)", fontsize=fs.SUPP_AXIS_FS)
    colorbar.ax.tick_params(labelsize=fs.SUPP_TICK_FS)


def draw_example_null(ax, matrix_data, example_null: pd.DataFrame) -> None:
    observed = abs(float(matrix_data["example_observed"]))
    source = int(matrix_data["example_source"])
    target = int(matrix_data["example_target"])
    nodes = matrix_data["unit_nodes"].astype(str)
    values = example_null["abs_NetTE"].to_numpy(float)
    ax.hist(values, bins=24, color="#BDBDBD", edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color="#C51B3A", lw=1.7)
    ax.text(0.97, 0.95, f"{nodes[source]} to {nodes[target]}\nillustrative pair",
            transform=ax.transAxes, ha="right", va="top", fontsize=fs.SUPP_SMALL_FS)
    ax.set_xlabel(r"Circular-shift $|\mathrm{NetTE}|$ (bits)")
    ax.set_ylabel("Count")
    clean_axis(ax)


def draw_observed_vs_null(ax, animal: pd.DataFrame, null: pd.DataFrame,
                          observed_column: str, null_column: str,
                          ylabel: str, chance_line: float | None = None) -> None:
    positions = np.arange(len(animal))
    labels = [value.replace("subject_", "S") for value in animal["recording_id"]]
    rng = np.random.default_rng(41)
    for position, (_, row) in zip(positions, animal.iterrows()):
        values = null.loc[null["recording_id"] == row["recording_id"], null_column].to_numpy(float)
        parts = ax.violinplot(values, positions=[position], widths=0.72,
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#BDBDBD")
            body.set_edgecolor("#666666")
            body.set_alpha(0.65)
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=2.0,
                   color="#777777", alpha=0.12, linewidth=0)
        ax.scatter(position, row[observed_column], s=28, color="#C51B3A",
                   edgecolor="black", linewidth=0.5, zorder=5)
    if chance_line is not None:
        ax.axhline(chance_line, color="#333333", ls="--", lw=0.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    if np.allclose(animal["global_p" if observed_column == "observed_global" else "exceedance_p"],
                   animal.iloc[0]["global_p" if observed_column == "observed_global" else "exceedance_p"]):
        p_column = "global_p" if observed_column == "observed_global" else "exceedance_p"
        ax.text(0.98, 0.96, f"all animals p = {animal.iloc[0][p_column]:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=fs.SUPP_SMALL_FS)
    clean_axis(ax)


def draw_region_heatmap(ax, region_animal: pd.DataFrame) -> None:
    canonical = pd.read_csv(REGION_FILE)
    canonical["division_order"] = pd.Categorical(
        canonical["anatomy_group"], categories=DIVISION_ORDER, ordered=True
    )
    canonical = canonical.sort_values(["division_order", "legacy_order"])
    animals = sorted(region_animal["recording_id"].unique())
    data = region_animal.copy()
    data["NetTE_z"] = data.groupby("recording_id")["NetTE"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    matrix = (
        data.pivot(index="node", columns="recording_id", values="NetTE_z")
        .reindex(index=canonical["node"], columns=animals)
    )
    image = ax.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                      aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(animals)), [value.replace("subject_", "S") for value in animals])
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=5.5)
    ax.set_xlabel("Animal")
    ax.set_ylabel("Anatomical region")
    previous = canonical.iloc[0]["anatomy_group"]
    for row, division in enumerate(canonical["anatomy_group"]):
        if row and division != previous:
            ax.axhline(row - 0.5, color="black", lw=0.5)
        previous = division
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("NetTE (within-animal z)", fontsize=fs.SUPP_AXIS_FS)
    colorbar.ax.tick_params(labelsize=fs.SUPP_TICK_FS)
    pairwise_r = []
    values = matrix.to_numpy(float)
    for first in range(values.shape[1]):
        for second in range(first + 1, values.shape[1]):
            pairwise_r.append(pearsonr(values[:, first], values[:, second]).statistic)
    ax.text(0.98, 1.02, f"mean animal-pair r = {np.mean(pairwise_r):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.SUPP_SMALL_FS)


def draw_loo(ax, loo: pd.DataFrame) -> None:
    full = loo.loc[loo["omitted_recording"] == "none_full_sample"].iloc[0]
    leaveout = loo.loc[loo["omitted_recording"] != "none_full_sample"].copy()
    leaveout["label"] = leaveout["omitted_recording"].str.replace("subject_", "S", regex=False)
    x = np.arange(len(leaveout))
    ax.axhline(full["r"], color="#C51B3A", lw=1.2,
               label=f"Full sample r = {full['r']:.2f}")
    ax.scatter(x, leaveout["r"], s=30, color="#2B6CB0", edgecolor="black", linewidth=0.5)
    ax.plot(x, leaveout["r"], color="#2B6CB0", lw=0.7, alpha=0.7)
    ax.axhline(0, color="#777777", lw=0.7, ls="--")
    ax.set_xticks(x, leaveout["label"])
    ax.set_xlabel("Omitted animal")
    ax.set_ylabel(r"FCV--NetTE Pearson $r$")
    ax.legend(frameon=False, fontsize=fs.SUPP_SMALL_FS, loc="upper right")
    clean_axis(ax)


def main() -> None:
    fs.apply_supplement_figure_style()
    animal = pd.read_csv(DATA_DIR / "te_surrogate_animal_summary.csv")
    null = pd.read_csv(DATA_DIR / "te_surrogate_network_null.csv")
    region_animal = pd.read_csv(DATA_DIR / "te_region_animal_values.csv")
    loo = pd.read_csv(DATA_DIR / "te_fcv_leave_one_animal_out.csv")
    example_null = pd.read_csv(DATA_DIR / "te_example_pair_null.csv")
    matrix_data = np.load(DATA_DIR / "te_example_observed_matrix.npz", allow_pickle=True)

    fig = plt.figure(figsize=(16.0, 8.0))
    fig.patch.set_facecolor("white")
    grid = GridSpec(2, 3, figure=fig, left=0.065, right=0.975, bottom=0.11,
                    top=0.94, wspace=0.38, hspace=0.38,
                    width_ratios=[1.08, 1.0, 1.05])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]

    draw_matrix(axes[0], matrix_data)
    draw_example_null(axes[1], matrix_data, example_null)
    draw_observed_vs_null(axes[2], animal, null, "observed_global", "mean_abs_NetTE",
                          r"Network mean $|\mathrm{NetTE}|$")
    draw_observed_vs_null(axes[3], animal, null, "observed_exceedance",
                          "extreme_pair_fraction", "Pairwise-null exceedance fraction",
                          chance_line=0.05)
    draw_region_heatmap(axes[4], region_animal)
    draw_loo(axes[5], loo)
    for ax, label in zip(axes, "ABCDEF"):
        panel_label(ax, label)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
