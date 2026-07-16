import os
import warnings
import sys
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Figure 12 workflow follows figure9_clean_v2.py:
# 1. Load data and prepare derived quantities for plotting.
# 2. Prepare the figure layout and axes.
# 3. Draw each panel.
# 4. Adjust panel positions and add panel labels.
# 5. Save figure files and statistics.

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.figure9_anatomy import ANATOMY_GROUP_ORDER

DEFAULT_REGIONS_FILE = (
    PROJECT_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)





def _parse_args():
    parser = argparse.ArgumentParser(description="Draw the clean Figure 12 panels")
    parser.add_argument(
        "--regions-file",
        type=Path,
        default=DEFAULT_REGIONS_FILE,
        help=(
            "CSV containing the node column to retain "
            f"(default: {DEFAULT_REGIONS_FILE})"
        ),
    )
    parser.add_argument("--output-suffix", default="", help="Suffix added to figure and statistics filenames")
    return parser.parse_args()


ARGS = _parse_args()
OUTPUT_SUFFIX = f"_{ARGS.output_suffix}" if ARGS.output_suffix else ""


STRUCTURAL_TABLE = "figure12_subject_region_structural_measures.csv"
ZF_GROUP_ORDER = list(ANATOMY_GROUP_ORDER)
ZF_GROUP_COLORS = fs.ZEBRAFISH_DIVISION_COLORS.copy()
PANEL_A_FEATURES = [
    ("OO\nfraction", "OO_fraction"),
    (r"$\mathbf{DCA}_{\mathbf{post}}$", "PostDCA"),
    (r"$\mathbf{DCA}_{\mathbf{pre}}$", "PreDCA"),
    ("Modularity\nQ", "Modularity"),
    ("log\n(Out/In)", "LogOutIn"),
]
PANEL_A_PRIORITY_REGIONS = ["SP", "rSP", "P", "rP", "PO", "rPO"]
PANEL_A_HIGHLIGHT_REGIONS = {"P", "rP", "SP",  "rPO", "PO", "rOB","rSP"}

fs.apply_main_figure_style()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
DATA = os.path.join(PROJECT_ROOT, "derived_data", "figure12")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "figures", f"figure12_final_v2{OUTPUT_SUFFIX}.png")
STATS_DIR = os.path.join(PROJECT_ROOT, "statistics")
STATS_CSV = os.path.join(STATS_DIR, f"figure12{OUTPUT_SUFFIX}_stats_v2.csv")
STATS_ROWS = []


# ============================================================
# Helpers copied from figure9_clean_v2.py style
# ============================================================
def _zscore(values):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / sd


def _fill_nan_with_mean(values):
    values = np.asarray(values, dtype=float)
    if not np.any(np.isnan(values)):
        return values
    out = values.copy()
    finite = np.isfinite(out)
    fill_value = float(np.nanmean(out[finite])) if np.any(finite) else 0.0
    out[~finite] = fill_value
    return out


def _reorder_linkage_tel_left_hind_right(z_linkage, divisions):
    """Flip dendrogram branches while preserving clustering topology."""
    reordered = np.array(z_linkage, copy=True)
    n_leaves = len(divisions)
    divisions = np.asarray(divisions).astype(str)
    stats_by_id = {}
    display_rank = {"Tel": 0.0, "Di": 1.0, "Mes": 2.0, "Hind": 5.0}
    for idx, div in enumerate(divisions):
        stats_by_id[idx] = {
            "n": 1,
            "tel": int(div == "Tel"),
            "hind": int(div == "Hind"),
            "rank_sum": float(display_rank.get(div, max(display_rank.values()) + 1.0)),
        }

    def _score(stats):
        n = max(int(stats["n"]), 1)
        mean_rank = stats["rank_sum"] / n
        hind_minus_tel = (stats["hind"] - stats["tel"]) / n
        return (mean_rank, hind_minus_tel)

    for row_idx, row in enumerate(reordered):
        left = int(row[0])
        right = int(row[1])
        parent = n_leaves + row_idx

        left_stats = stats_by_id[left]
        right_stats = stats_by_id[right]
        if _score(right_stats) < _score(left_stats):
            reordered[row_idx, 0], reordered[row_idx, 1] = reordered[row_idx, 1], reordered[row_idx, 0]
            left, right = right, left
            left_stats, right_stats = right_stats, left_stats

        stats_by_id[parent] = {
            "n": left_stats["n"] + right_stats["n"],
            "tel": left_stats["tel"] + right_stats["tel"],
            "hind": left_stats["hind"] + right_stats["hind"],
            "rank_sum": left_stats["rank_sum"] + right_stats["rank_sum"],
        }

    return reordered


def _move_last_cluster_run_to_middle(z_linkage, leaf_order, n_clusters=3):
    """Cut leaves into contiguous cluster runs and move the last run to middle."""
    leaf_order = np.asarray(leaf_order, dtype=int)
    cluster_labels = fcluster(z_linkage, t=n_clusters, criterion="maxclust")
    runs = []
    start = 0
    current_label = cluster_labels[leaf_order[0]]
    for pos, leaf_idx in enumerate(leaf_order[1:], start=1):
        label = cluster_labels[leaf_idx]
        if label != current_label:
            runs.append(leaf_order[start:pos])
            start = pos
            current_label = label
    runs.append(leaf_order[start:])
    if len(runs) != n_clusters:
        return leaf_order
    return np.concatenate([runs[0], runs[-1], *runs[1:-1]])


def _move_priority_cluster_left(z_linkage, leaf_order, region_names, priority_names, n_clusters=3):
    """Place the full cluster containing priority regions first."""
    leaf_order = np.asarray(leaf_order, dtype=int)
    region_names = np.asarray(region_names).astype(str)
    priority_names = set(str(name) for name in priority_names)
    cluster_labels = fcluster(z_linkage, t=n_clusters, criterion="maxclust")
    priority_clusters = {
        int(cluster_labels[idx])
        for idx, name in enumerate(region_names)
        if name in priority_names
    }
    if not priority_clusters:
        return leaf_order
    left = [idx for idx in leaf_order if int(cluster_labels[idx]) in priority_clusters]
    rest = [idx for idx in leaf_order if int(cluster_labels[idx]) not in priority_clusters]
    return np.array(left + rest, dtype=int)


def _add_figure12_panel_labels(ax_dendro, ax_b, ax_c, ax_d):
    ax_dendro.text(
        -0.090, 1.5, "A",
        transform=ax_dendro.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    for ax, label in [
        (ax_b, "B"),
        (ax_c, "C"),
    ]:
        ax.text(
            -0.40, 1.05, label,
            transform=ax.transAxes,
            fontsize=fs.PANEL_LABEL_FS_2COL,
            fontweight="bold",
            va="bottom",
        )
        
    for ax, label in [
        (ax_d, "D"),
    ]:
        ax.text(
            -0.31, 1.05, label,
            transform=ax.transAxes,
            fontsize=fs.PANEL_LABEL_FS_2COL,
            fontweight="bold",
            va="bottom",
        )


def _load_latest_zebrafish_sc_values():
    out = pd.read_csv(os.path.join(DATA, STRUCTURAL_TABLE))
    out = out.loc[out["species"].eq("Zebrafish")].copy()
    if ARGS.regions_file:
        region_table = pd.read_csv(ARGS.regions_file)
        if "node" not in region_table.columns:
            raise KeyError(f"Region filter lacks a node column: {ARGS.regions_file}")
        selected_nodes = set(region_table["node"].astype(str))
        missing_nodes = selected_nodes.difference(set(out["node"].astype(str)))
        if missing_nodes:
            raise ValueError(f"Figure 12 input lacks selected nodes: {sorted(missing_nodes)}")
        out = out.loc[out["node"].astype(str).isin(selected_nodes)].copy()
        print(f"Figure 12 region filter: {out['node'].nunique()} nodes, {len(out)} rows")
    out = out.replace([np.inf, -np.inf], np.nan)
    out["_group_order"] = out["anatomy_group"].map(
        {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
    ).fillna(len(ZF_GROUP_ORDER)).astype(int)
    return out.sort_values(["_group_order", "anatomy_group", "node"]).reset_index(drop=True)


def _wide_division_df(plot_df, measure):
    plot_df = plot_df[["anatomy_group", measure]].dropna().copy()
    grouped = {
        div: plot_df.loc[plot_df["anatomy_group"] == div, measure].to_numpy()
        for div in ZF_GROUP_ORDER
    }
    max_len = max((len(v) for v in grouped.values()), default=0)
    return pd.DataFrame({
        div: pd.Series(vals, dtype=float).reindex(range(max_len))
        for div, vals in grouped.items()
    })


# ============================================================
# 1. Data loading and plotting calculations
# ============================================================
sc_values = _load_latest_zebrafish_sc_values()
panel_a_cols = [col for _, col in PANEL_A_FEATURES]

# Panel A: average raw recording-node values at the node level first, then
# z-score each measure across nodes (average-then-zscore).
fig12_zf = (
    sc_values.groupby(["node", "anatomy_group"], as_index=False)
    .agg(**{col: (col, "mean") for col in panel_a_cols})
    .dropna(subset=panel_a_cols, how="any")
)
for col in panel_a_cols:
    fig12_zf[col] = _zscore(fig12_zf[col].to_numpy(float))
fig12_zf["_group_order"] = fig12_zf["anatomy_group"].map(
    {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
).fillna(len(ZF_GROUP_ORDER)).astype(int)
fig12_zf = fig12_zf.sort_values(["_group_order", "anatomy_group", "node"]).reset_index(drop=True)

regions_d = fig12_zf["node"].astype(str).to_numpy()
divs_d = fig12_zf["anatomy_group"].astype(str).to_numpy()
out_data = fig12_zf[panel_a_cols].to_numpy(float).T

div_color_map = {
    **ZF_GROUP_COLORS,
}

# Panel A preserves Ward clustering of measured SC features, but flips
# dendrogram branches so clusters enriched for Tel tend to appear on the left.
Z = linkage(out_data.T, method="ward")
Z = _reorder_linkage_tel_left_hind_right(Z, divs_d)

dend_info = dendrogram(Z, no_plot=True)
leaf_order = np.array(dend_info["leaves"])
leaf_order = _move_last_cluster_run_to_middle(Z, leaf_order, n_clusters=3)
leaf_order = _move_priority_cluster_left(
    Z,
    leaf_order,
    regions_d,
    PANEL_A_PRIORITY_REGIONS,
    n_clusters=3,
)

out_data_c = out_data[:, leaf_order]
regions_c = regions_d[leaf_order]
divs_c = divs_d[leaf_order]
n_regions = out_data_c.shape[1]
y_labels = [label for label, _ in PANEL_A_FEATURES]


# ============================================================
# 2. Layout preparation
# ============================================================
# Top=[A (dendro+divbar+heat)] | Bottom=[B C D]
_fig_w = fs.MAIN_FIGURE_WIDTH
_fig_h = fs.MAIN_FIGURE_HEIGHT_TALL
fig = plt.figure(figsize=(_fig_w, _fig_h))
gs = GridSpec(
    5, 12,
    figure=fig,
    height_ratios=[0.8, 0.16, 1.35, 1.7, 1.7],
    width_ratios=[1] * 12,
    left=0.08, right=0.97,
    top=0.94, bottom=0.20,
    hspace=0.24, wspace=0.34,
)

# Panel A axes: same structure as figure9_clean_v2.py
ax_dendro = fig.add_subplot(gs[0, 0:10])
ax_divbar_a = fig.add_subplot(gs[1, 0:10])
ax_heat = fig.add_subplot(gs[2, 0:10])
ax_cbar = fig.add_subplot(gs[2, 11:12])

# Bottom panels
ax_b = fig.add_subplot(gs[3:5, 0:4])
ax_c = fig.add_subplot(gs[3:5, 4:8])
ax_d = fig.add_subplot(gs[3:5, 8:12])


division_colors = ZF_GROUP_COLORS
division_order = [group for group in ZF_GROUP_ORDER if group in set(sc_values["anatomy_group"])]
division_short_labels = {
    "Tel": "Tel",
    "Di": "Di",
    "Mes": "Mes",
    "Hind": "Hind",
}
_STAR_FS = fs.STAR_FS_2COL


def _record_division_stats(panel, df, order):
    groups = [df[k].dropna().values for k in order]
    global_test = kruskal(*groups)
    STATS_ROWS.append({
        "figure": "figure12",
        "panel": panel,
        "test": "Kruskal-Wallis",
        "alternative": "two-sided",
        "group_1": "all divisions",
        "group_2": "",
        "n_group_1": int(sum(len(group) for group in groups)),
        "n_group_2": np.nan,
        "statistic": float(global_test.statistic),
        "p_uncorrected": float(global_test.pvalue),
        "p_holm": np.nan,
        "reject_holm_0.05": bool(global_test.pvalue < 0.05),
    })
    pairs = list(itertools.combinations(range(len(order)), 2))
    pvals = [
        mannwhitneyu(groups[i], groups[j], alternative="two-sided")[1]
        for i, j in pairs
    ]
    reject, corr, _, _ = multipletests(pvals, method="holm")
    for idx, (i, j) in enumerate(pairs):
        STATS_ROWS.append({
            "figure": "figure12",
            "panel": panel,
            "test": "Mann-Whitney U",
            "alternative": "two-sided",
            "group_1": order[i],
            "group_2": order[j],
            "n_group_1": len(groups[i]),
            "n_group_2": len(groups[j]),
            "p_uncorrected": pvals[idx],
            "p_holm": corr[idx],
            "reject_holm_0.05": bool(reject[idx]),
        })
    return pairs, reject, corr


def _add_sig_bars(ax, df, order, panel):
    pairs, reject, corr = _record_division_stats(panel, df, order)
    sig = sorted(
        [(pairs[k], corr[k]) for k in range(len(pairs)) if reject[k]],
        key=lambda x: x[0][1] - x[0][0],
    )
    if not sig:
        return
    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    step = yr * 0.090
    bar_h = yr * 0.018
    for lvl, ((i, j), p) in enumerate(sig):
        y = y_max + yr * 0.025 + lvl * step
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.plot(
            [i, i, j, j],
            [y, y + bar_h, y + bar_h, y],
            lw=0.65,
            c="#333",
            clip_on=False,
        )
        ax.text(
            (i + j) / 2, y + bar_h, star,
            ha="center", va="center", fontsize=_STAR_FS,
            clip_on=False,
        )
    ax.set_ylim(y_min, y_max)


def _boxplot_panel(ax, df, ylabel, ylim=None):
    vals, labels = [], []
    for col in division_order:
        v = df[col].dropna().values
        vals.extend(v)
        labels.extend([col] * len(v))
    fs.draw_main_box_strip(
        ax,
        labels,
        vals,
        division_order,
        palette=[division_colors[d] for d in division_order],
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    ax.set_xticks(range(len(division_order)))
    ax.set_xticklabels(
        [division_short_labels.get(group, group) for group in division_order],
        rotation=0,
        ha="center",
    )
    if ylim is not None:
        ax.set_ylim(*ylim)


def _format_dca_axis(ax):
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.yaxis.labelpad = 6


# Figure 12 zebrafish data (Panels B-D). Standardize regions within each
# subject and retain every subject-region observation as an individual point.
panel_bcd_measures = ["PostDCA", "PreDCA", "OO_fraction"]
sc_values_subject_z = sc_values.copy()
for _col in panel_bcd_measures:
    sc_values_subject_z[_col] = sc_values_subject_z.groupby("recording_id")[_col].transform(
        lambda values: _zscore(values.to_numpy(dtype=float))
    )

_post_dca = _wide_division_df(sc_values_subject_z, "PostDCA")
_pre_dca = _wide_division_df(sc_values_subject_z, "PreDCA")
_oo_fraction = _wide_division_df(sc_values_subject_z, "OO_fraction")



_boxplot_panel(ax_b, _post_dca, r"$\mathrm{DCA}_{\mathrm{post}}$")
_boxplot_panel(ax_c, _pre_dca, r"$\mathrm{DCA}_{\mathrm{pre}}$")
_boxplot_panel(ax_d, _oo_fraction, "OO fraction")


_add_sig_bars(ax_b, {c: _post_dca[c] for c in division_order}, division_order, r"$\mathrm{DCA}_{\mathrm{post}}$")
_add_sig_bars(ax_c, {c: _pre_dca[c] for c in division_order}, division_order, r"$\mathrm{DCA}_{\mathrm{pre}}$")
_add_sig_bars(ax_d, {c: _oo_fraction[c] for c in division_order}, division_order, "OO fraction")
_format_dca_axis(ax_b)
_format_dca_axis(ax_c)

# Panel A: dendrogram, division bar, and feature heatmap
dendrogram(
    Z,
    ax=ax_dendro,
    no_labels=True,
    color_threshold=0,
    above_threshold_color="#333333",
)
ax_dendro.set_xlim(0, n_regions * 10)
dend_max_h = max(max(d) for d in dend_info["dcoord"])
ax_dendro.set_ylim(0, dend_max_h * 1.05)
ax_dendro.axis("off")

for i, div in enumerate(divs_c):
    ax_divbar_a.add_patch(
        plt.Rectangle((i - 0.5, 0), 1, 1, color=div_color_map.get(div, "gray"), linewidth=0)
    )

ax_divbar_a.set_xlim(-0.5, n_regions - 0.5)
ax_divbar_a.set_ylim(0, 1)
ax_divbar_a.set_xticks([])
ax_divbar_a.set_yticks([])
for spine in ax_divbar_a.spines.values():
    spine.set_visible(False)

im = ax_heat.imshow(
    out_data_c,
    aspect="auto",
    cmap="RdBu_r",
    vmax=2,
    vmin=-2,
    interpolation="nearest",
)
ax_heat.set_xlim(-0.5, n_regions - 0.5)

ax_heat.set_yticks(range(len(y_labels)))
ax_heat.set_yticklabels(y_labels, fontsize=fs.TICK_FS_2COL - 2)
ax_heat.tick_params(axis="y", length=0, labelright=False, labelleft=True)

ax_heat.set_xticks(range(n_regions))
ax_heat.set_xticklabels(
    regions_c,
    rotation=90,
    fontsize=fs.TICK_FS_2COL - 2,
    ha="center",
)
ax_heat.tick_params(axis="x", length=2, width=0.5)

for tick_label, div in zip(ax_heat.get_xticklabels(), divs_c):
    tick_label.set_color(div_color_map[div])

for y in np.arange(-0.5, len(y_labels), 1):
    ax_heat.axhline(y=y, color="white", linewidth=0.4)

_highlight_idx = [i for i, name in enumerate(regions_c) if str(name) in PANEL_A_HIGHLIGHT_REGIONS]
if _highlight_idx:
    _x0 = min(_highlight_idx) - 0.5
    _width = max(_highlight_idx) - min(_highlight_idx) + 1
    _highlight_edge = "#0AEB60FF"
    ax_divbar_a.add_patch(
        mpatches.Rectangle(
            (_x0, 0),
            _width,
            1,
            fill=False,
            edgecolor=_highlight_edge,
            linewidth=1.8,
            zorder=20,
            clip_on=False,
        )
    )
    ax_heat.add_patch(
        mpatches.Rectangle(
            (_x0, -0.5),
            _width,
            len(y_labels),
            fill=False,
            edgecolor=_highlight_edge,
            linewidth=1.8,
            zorder=25,
            clip_on=False,
        )
    )

for spine in ax_heat.spines.values():
    spine.set_linewidth(1.4)
    spine.set_color("black")
ax_heat.add_patch(
    mpatches.Rectangle(
        (-0.5, -0.5), n_regions, len(y_labels),
        fill=False,
        edgecolor="black",
        linewidth=1.4,
        zorder=10,
        clip_on=False,
    )
)

cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.set_title("Z-score", fontsize=fs.AXIS_LABEL_FS_2COL)
cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)
cbar.set_ticks([-2, -1, 0, 1, 2])

div_handles = [mpatches.Patch(color=div_color_map[d], label=d) for d in division_order]
ax_heat.legend(
    handles=div_handles,
    loc="upper left",
    bbox_to_anchor=(0.4, 2.0),
    ncol=4,
    fontsize=fs.TICK_FS_2COL,
    frameon=False,
    framealpha=0.95,
    title_fontsize=fs.AXIS_LABEL_FS_2COL,
)


# ============================================================
# 4. Panel position adjustment and panel labels
# ============================================================
_panel_a_dy = 0.1
_panel_a_dx = 0.06
for _ax in [ax_dendro, ax_divbar_a, ax_heat, ax_cbar]:
    _p = _ax.get_position()
    _ax.set_position([_p.x0 + _panel_a_dx, _p.y0 + _panel_a_dy, _p.width, _p.height])

for _ax in [ax_heat]:
    _p = _ax.get_position()
    _ax.set_position([_p.x0 - _p.width * 0.045, _p.y0 - _p.height * 0.5, _p.width * 1.045, _p.height * 1.5])

for _ax in [ax_dendro, ax_divbar_a]:
    _p = _ax.get_position()
    _ax.set_position([_p.x0 - _p.width * 0.045, _p.y0, _p.width * 1.045, _p.height])

_cb_pos = ax_cbar.get_position()
ax_cbar.set_position([_cb_pos.x0, _cb_pos.y0, _cb_pos.width * 0.4, _cb_pos.height * 1.5])

_cb_pos = ax_cbar.get_position()
ax_cbar.set_position([_cb_pos.x0 - 0.07, _cb_pos.y0 - 0.0715, _cb_pos.width, _cb_pos.height])

for _ax in [ax_b, ax_c, ax_d]:
    _p = _ax.get_position()
    _cx = _p.x0 + _p.width / 2
    _cy = _p.y0 + _p.height / 2
    _nw = _p.width * 0.78
    _nh = _p.height * 0.50
    _ax.set_position([_cx - _nw / 2, _cy - _nh / 2, _nw, _nh])

_add_figure12_panel_labels(ax_dendro, ax_b, ax_c, ax_d)


# ============================================================
# 5. Save figure and statistics
# ============================================================
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", transparent=False)
os.makedirs(STATS_DIR, exist_ok=True)
pd.DataFrame(STATS_ROWS).to_csv(STATS_CSV, index=False)
print(f"Saved {STATS_CSV}")
