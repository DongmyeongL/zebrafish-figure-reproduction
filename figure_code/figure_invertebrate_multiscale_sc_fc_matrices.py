"""Plot multiscale invertebrate structural and functional matrices."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import figure_style as fs


PACK = Path(__file__).resolve().parents[1]
DATA = PACK / "derived_data" / "invertebrates" / "invertebrate_multiscale_sc_fc_matrices.npz"
OUT = PACK / "figures" / "figure_invertebrate_multiscale_sc_fc_matrices.png"

GROUP_COLORS = {
    "chemosensory": "#6FA8C9",
    "mechanosensory / other sensory": "#4E9DA6",
    "thermo/gas sensory": "#DDA15E",
    "interneuron / integrative": "#6FAF6B",
    "associative interneuron": "#8B7CB6",
    "state-modulatory interneuron": "#E76F51",
    "head motor / premotor": "#D48AB0",
    "locomotor command interneuron": "#C9884B",
    "olfactory-associative": "#6FA8C9",
    "visual-association / integrative": "#6FAF6B",
    "premotor / descending": "#D48AB0",
    "primary visual": "#DDA15E",
    "lateral/inferior protocerebrum": "#8B7CB6",
    "other / central complex": "#909090",
}
SHORT_GROUP = {
    "chemosensory": "chemo",
    "mechanosensory / other sensory": "mechano",
    "thermo/gas sensory": "thermo/gas",
    "interneuron / integrative": "inter/integ",
    "associative interneuron": "assoc. inter",
    "state-modulatory interneuron": "state inter",
    "head motor / premotor": "head motor",
    "locomotor command interneuron": "loco. cmd",
    "olfactory-associative": "olf.-assoc.",
    "visual-association / integrative": "vis.-assoc./integ.",
    "premotor / descending": "premotor",
    "primary visual": "primary visual",
    "lateral/inferior protocerebrum": "lat/inf proto",
    "other / central complex": "other/CX",
}


def _runs(groups: np.ndarray) -> list[tuple[str, int, int]]:
    groups = np.asarray(groups).astype(str)
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    ends = np.r_[starts[1:], len(groups)]
    return [(groups[start], int(start), int(end)) for start, end in zip(starts, ends)]


def _decorate_matrix(ax, groups: np.ndarray, show_group_names: bool = True) -> None:
    n = len(groups)
    strip_offset = -max(0.8, 0.012 * n)
    label_offset = -max(2.0, 0.040 * n)
    for group, start, end in _runs(groups):
        if start:
            ax.axvline(start - 0.5, color="white", linewidth=0.65, alpha=0.85)
            ax.axhline(start - 0.5, color="white", linewidth=0.65, alpha=0.85)
        color = GROUP_COLORS.get(group, "#909090")
        ax.plot(
            [start - 0.45, end - 0.55], [strip_offset, strip_offset],
            color=color, lw=5, clip_on=False,
        )
        ax.plot(
            [strip_offset, strip_offset], [start - 0.45, end - 0.55],
            color=color, lw=5, clip_on=False,
        )
        if show_group_names:
            ax.text(
                (start + end - 1) / 2,
                label_offset,
                SHORT_GROUP.get(group, group),
                color=color,
                fontsize=6.5,
                fontweight="bold",
                rotation=32,
                ha="left",
                va="bottom",
                clip_on=False,
            )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Target", labelpad=3)
    ax.set_ylabel("Source", labelpad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def _plot_sc(ax, density: np.ndarray, groups: np.ndarray, title: str):
    shown = np.log10(1.0 + 1e6 * np.nan_to_num(density, nan=0.0))
    positive = shown[shown > 0]
    vmax = np.percentile(positive, 99.5) if positive.size else 1.0
    image = ax.imshow(shown, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest", rasterized=True)
    _decorate_matrix(ax, groups)
    ax.set_title(title, fontsize=fs.SUPP_TITLE_FS, fontweight="bold", y=1.16, pad=0)
    return image


def _plot_ce_sc(ax, weights: np.ndarray, groups: np.ndarray):
    shown = np.log10(1.0 + weights)
    positive = shown[shown > 0]
    vmax = np.percentile(positive, 99.5) if positive.size else 1.0
    image = ax.imshow(shown, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest", rasterized=True)
    _decorate_matrix(ax, groups)
    ax.set_title(
        "Neuron-level chemical-synapse SC",
        fontsize=fs.SUPP_TITLE_FS,
        fontweight="bold",
        y=1.16,
        pad=0,
    )
    return image


def _plot_fc(ax, fc: np.ndarray, groups: np.ndarray, title: str, vmax: float):
    image = ax.imshow(
        fc,
        cmap="RdBu_r",
        norm=Normalize(vmin=-vmax, vmax=vmax),
        interpolation="nearest",
        rasterized=True,
    )
    _decorate_matrix(ax, groups)
    ax.set_title(title, fontsize=fs.SUPP_TITLE_FS, fontweight="bold", y=1.16, pad=0)
    return image


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=fs.SUPP_PANEL_LABEL_FS,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def main() -> None:
    fs.apply_supplement_figure_style()
    with np.load(DATA) as data:
        arrays = {key: data[key] for key in data.files}

    fc_values = np.concatenate([
        np.abs(arrays["ce_fc"][np.isfinite(arrays["ce_fc"])]),
        np.abs(arrays["fly_fc"][np.isfinite(arrays["fly_fc"])]),
    ])
    fc_vmax = max(0.25, float(np.percentile(fc_values, 99.0)))

    fig = plt.figure(figsize=(16.0, 9.6))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.05],
        left=0.055,
        right=0.975,
        bottom=0.065,
        top=0.925,
        hspace=0.46,
    )
    top = outer[0].subgridspec(1, 2, wspace=0.24)
    bottom = outer[1].subgridspec(1, 3, wspace=0.30)
    axes = {
        "A": fig.add_subplot(top[0, 0]),
        "B": fig.add_subplot(top[0, 1]),
        "C": fig.add_subplot(bottom[0, 0]),
        "D": fig.add_subplot(bottom[0, 1]),
        "E": fig.add_subplot(bottom[0, 2]),
    }

    images = {}
    images["A"] = _plot_ce_sc(axes["A"], arrays["ce_sc"], arrays["ce_groups"])
    images["B"] = _plot_fc(
        axes["B"], arrays["ce_fc"], arrays["ce_groups"], "Mean neuron-level FC", fc_vmax
    )
    images["C"] = _plot_sc(
        axes["C"],
        arrays["fly_cell_sc_density"],
        arrays["fly_cell_bin_groups"],
        "Cellular SC (256 display bins)",
    )
    images["D"] = _plot_sc(
        axes["D"],
        arrays["fly_subunit_sc_density"],
        arrays["fly_subunit_groups"],
        "137-unit region-level SC",
    )
    images["E"] = _plot_fc(
        axes["E"], arrays["fly_fc"], arrays["fly_region_groups"], "Mean 41-region FC", fc_vmax
    )

    for label, ax in axes.items():
        _panel_label(ax, label)

    fig.canvas.draw()
    top_pos = [axes[label].get_position() for label in "AB"]
    bottom_pos = [axes[label].get_position() for label in "CDE"]
    fig.text(
        (top_pos[0].x0 + top_pos[-1].x1) / 2,
        max(pos.y1 for pos in top_pos) + 0.072,
        "C. elegans",
        ha="center",
        va="bottom",
        fontsize=fs.SUPP_TITLE_FS + 1,
        fontweight="bold",
        fontstyle="italic",
    )
    fig.text(
        (bottom_pos[0].x0 + bottom_pos[-1].x1) / 2,
        max(pos.y1 for pos in bottom_pos) + 0.072,
        "Drosophila",
        ha="center",
        va="bottom",
        fontsize=fs.SUPP_TITLE_FS + 1,
        fontweight="bold",
        fontstyle="italic",
    )

    for label in ("A", "C", "D"):
        cbar = fig.colorbar(images[label], ax=axes[label], fraction=0.038, pad=0.028)
        cbar.set_label(
            r"$\log_{10}(1 + 10^6 \times \mathrm{density})$" if label != "A"
            else r"$\log_{10}(1 + \mathrm{synapse\ count})$",
            fontsize=7.5,
        )
        cbar.ax.tick_params(labelsize=7)
    for label in ("B", "E"):
        cbar = fig.colorbar(images[label], ax=axes[label], fraction=0.038, pad=0.028)
        cbar.set_label("Pearson correlation", fontsize=7.5)
        cbar.ax.tick_params(labelsize=7)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
