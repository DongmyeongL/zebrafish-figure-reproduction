import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu

import figure_style as fs
fs.apply_supplement_figure_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "raw_data" / "figure_supply_5"
NETWORK_DIR = RAW_DATA / "network_diagrams"
STATS_DIR = RAW_DATA / "null_statistics"
OUT_STATS = PROJECT_ROOT / "statistics" / "figure_supply_5_stats.csv"
OUT_PNG = PROJECT_ROOT / "figures" / "figure_supply_5.png"

PAIR_SPECS = [
    {
        "label": "",
        "area_labels": ("P", "imRF"),
        "base": "network_rP_rimRF_network_diagarm.png",
        "null_in": "network_null_in_rP_rimRF_network_diagarm.png",
        "null_out": "network_null_out_rP_rimRF_network_diagarm.png",
    },
    {
        "label": "",
        "area_labels": ("P", "MOS4"),
        "base": "network_rP_rMOS4_network_diagarm.png",
        "null_in": "network_null_in_rP_rMOS4_network_diagarm.png",
        "null_out": "network_null_out_rP_rMOS4_network_diagarm.png",
    },
]

COLUMNS = [
    ("base", "Original"),
    ("null_in", "Null-In"),
    ("null_out", "Null-Out"),
]


def p_text(p_value):
    return f"p = {p_value:.3g}" if p_value >= 0.001 else "p < 0.001"


def read_network_image(filename):
    path = NETWORK_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def plot_null_box(ax, npz_name, ylabel, xticklabels, panel_label):
    data = np.load(STATS_DIR / npz_name)
    base = -1 * np.asarray(data["x_data"], dtype=float)
    null = -1 * np.asarray(data["y_data"], dtype=float)
    base = base[np.isfinite(base)]
    null = null[np.isfinite(null)]
    test_result = mannwhitneyu(base, null, alternative="two-sided")
    p_value = float(test_result.pvalue)

    bp = ax.boxplot(
        [base, null],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], ["#6baed6", "#fdae6b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for idx, values in enumerate([base, null], start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=8,
            color="black",
            alpha=0.38,
            linewidth=0,
            zorder=3,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.text(
        0.98,
        0.94,
        p_text(p_value),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fs.SUPP_SMALL_FS,
        fontstyle="italic",
    )
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1.0)
    return {
        "figure": "figure_supply_5",
        "panel": panel_label,
        "metric": ylabel,
        "test": "Mann-Whitney U",
        "alternative": "two-sided",
        "comparison": f"{xticklabels[0]} vs {xticklabels[1]}",
        "source_npz": f"raw_data/figure_supply_5/null_statistics/{npz_name}",
        "n_base": int(len(base)),
        "n_null": int(len(null)),
        "mean_base": float(np.mean(base)) if len(base) else np.nan,
        "mean_null": float(np.mean(null)) if len(null) else np.nan,
        "median_base": float(np.median(base)) if len(base) else np.nan,
        "median_null": float(np.median(null)) if len(null) else np.nan,
        "u_statistic": float(test_result.statistic),
        "p_value": p_value,
        "p_text": p_text(p_value),
    }


def add_panel_label(ax, label):
    ax.text(
        -0.05,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=fs.SUPP_PANEL_LABEL_FS,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def shrink_axis_width(ax, width_scale=0.70):
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    new_w = pos.width * width_scale
    ax.set_position([cx - new_w / 2, pos.y0, new_w, pos.height])


def shift_axes_down(axes, dy=0.030):
    for ax in axes.ravel():
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - dy, pos.width, pos.height])


def add_network_area_labels(ax, top_label, bottom_label):
    ax.text(
        0.50,
        0.90,
        top_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs.SUPP_AXIS_FS,
        fontweight="bold",
        color="#333333",
    )
    ax.text(
        0.50,
        0.10,
        bottom_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fs.SUPP_AXIS_FS,
        fontweight="bold",
        color="#333333",
    )


def add_dca_colorbar_with_arrows(fig, axes):
    positions = [ax.get_position() for ax in axes.ravel()]
    x0 = min(pos.x0 for pos in positions)
    x1 = max(pos.x1 for pos in positions)
    y0 = min(pos.y0 for pos in positions)

    cbar_ax = fig.add_axes([x0 + 0.31 * (x1 - x0), y0 - 0.035, 0.38 * (x1 - x0), 0.010])
    cbar_mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="jet")
    cbar_mappable.set_array([])
    cbar = fig.colorbar(cbar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=fs.SUPP_SMALL_FS, length=2, pad=1)
    cbar.set_label(r"$\mathrm{DCA}$", fontsize=fs.SUPP_AXIS_FS, labelpad=0.1)
    cbar_ax.annotate(
        "",
        xy=(1.18, 2.35),
        xytext=(1.18, -1.25),
        xycoords=cbar_ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#e5a247", lw=2.4, mutation_scale=12),
        clip_on=False,
    )
    cbar_ax.annotate(
        "",
        xy=(1.34, -2.02),
        xytext=(1.34, 1.58),
        xycoords=cbar_ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#8bcef4", lw=2.4, mutation_scale=12),
        clip_on=False,
    )


def make_figure():
    fig = plt.figure(figsize=(16.0, 13.08))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.85, 1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=0.055,
        wspace=0.25,
        hspace=0.30,
    )

    ax_stat_out = fig.add_subplot(gs[0, 0:3])
    ax_stat_in = fig.add_subplot(gs[0, 3:6])
    stats_rows = []
    stats_rows.append(plot_null_box(
        ax_stat_out,
        "figure6_network_properites_in.npz",
        r"$\mathrm{DCA}_{\mathrm{post}}$",
        ["Base", "Null-Out"],
        "A",
    ))
    stats_rows.append(plot_null_box(
        ax_stat_in,
        "figure6_network_properites_out.npz",
        r"$\mathrm{DCA}_{\mathrm{pre}}$",
        ["Base", "Null-In"],
        "B",
    ))
    shrink_axis_width(ax_stat_out, width_scale=0.70)
    shrink_axis_width(ax_stat_in, width_scale=0.70)

    network_axes = np.empty((len(PAIR_SPECS), len(COLUMNS)), dtype=object)
    for row_idx in range(len(PAIR_SPECS)):
        for col_idx in range(len(COLUMNS)):
            network_axes[row_idx, col_idx] = fig.add_subplot(
                gs[row_idx + 1, col_idx * 2:(col_idx + 1) * 2]
            )

    add_panel_label(ax_stat_out, "A")
    add_panel_label(ax_stat_in, "B")
    for row_idx, pair in enumerate(PAIR_SPECS):
        for col_idx, (key, col_title) in enumerate(COLUMNS):
            ax = network_axes[row_idx, col_idx]
            image = read_network_image(pair[key])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            add_network_area_labels(ax, *pair["area_labels"])

            if row_idx == 0:
                ax.set_title(col_title, fontsize=fs.SUPP_TITLE_FS, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    pair["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=fs.SUPP_TITLE_FS,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
            if col_idx == 0:
                add_panel_label(ax, "C" if row_idx == 0 else "D")

    shift_axes_down(network_axes, dy=0.030)
    add_dca_colorbar_with_arrows(fig, network_axes)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_csv(OUT_STATS, index=False)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def main():
    make_figure()
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
