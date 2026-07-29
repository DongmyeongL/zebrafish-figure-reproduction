import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Figure 9 workflow:
# 1. Load data and prepare derived quantities for plotting.
# 2. Prepare the figure layout and axes.
# 3. Draw each panel.
# 4. Adjust panel positions and add panel labels.
# 5. Save figure files and statistics.

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox
import matplotlib.patches as mpatches

import figure_style as fs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.figure9_anatomy import ANATOMY_GROUP_ORDER, anatomy_group

DEFAULT_REGIONS_FILE = (
    PROJECT_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Draw the clean Figure 9 panels")
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


FIG1_RECORDING_MEASURE_TABLE = "figure9_recording_region_measures.csv"
ZF_GROUP_ORDER = list(ANATOMY_GROUP_ORDER)
ZF_GROUP_COLORS = fs.ZEBRAFISH_DIVISION_COLORS.copy()
MEASURE_COLUMNS = [
    "EdgeStdFCV",
    "FCS",
    "ProfileCorrDistFCV",
    "ObservedNetTE",
    "NeighborNetTE",
]
PANEL_A_FEATURES = [
    ("FCV", "EdgeStdFCV"),
    ("FCS", "FCS"),
    ("FC\nRecon.", "ProfileCorrDistFCV"),
    (r"$\mathbf{TE}_{\mathbf{net}}$", "ObservedNetTE"),
    ("Neigh.\n" + r"$\mathbf{TE}_{\mathbf{net}}$", "NeighborNetTE"),
]
PANEL_A_PRIORITY_REGIONS = ["SP", "rSP", "P", "rP", "PO", "rPO"]
PANEL_A_HIGHLIGHT_REGIONS = {"P", "rP", "SP", "rSP", "rOB","PO", "rPO"}

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
DATA = os.path.join(PROJECT_ROOT, "derived_data", "figure9")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "figures", f"figure9_final_v2{OUTPUT_SUFFIX}.png")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "figures", f"figure9_final_v2{OUTPUT_SUFFIX}.pdf")
STATS_DIR = os.path.join(PROJECT_ROOT, "statistics")
STATS_CSV = os.path.join(STATS_DIR, f"figure9_final_v2{OUTPUT_SUFFIX}_stats.csv")
STATS_ROWS = []
PLOT_RANDOM_SEED = 42
np.random.seed(PLOT_RANDOM_SEED)


# ============================================================
# Helpers
# ============================================================
def _zscore(values):
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / sd


def _reorder_linkage_tel_left_hind_right(z_linkage, divisions):
    """Flip dendrogram branches while preserving clustering topology.

    Branches enriched for Tel are placed left when possible, and branches
    enriched for Hind are placed right. This does not force division blocks;
    it only chooses the left/right orientation of existing dendrogram branches.
    """
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
        # Smaller scores are placed to the left. Mean division rank encourages
        # the left-to-right progression Tel -> Di -> Mes -> Hind, so Mes-rich
        # branches tend to stay between forebrain and hindbrain clusters while
        # preserving the original Ward clustering topology.
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


def _add_figure9_panel_labels(ax_dendro, ax_c, ax_d, ax_e):
    ax_dendro.text(
        -0.065, 1.5, 'A',
        transform=ax_dendro.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight='bold',
        va='bottom',
        ha='right',
    )

    for ax, label in [
        (ax_c, 'B'),
        (ax_d, 'C'),
        (ax_e, 'D'),
    ]:
        ax.text(
            -0.31, 1.05, label,
            transform=ax.transAxes,
            fontsize=fs.PANEL_LABEL_FS_2COL,
            fontweight='bold',
            va='bottom',
        )

# ============================================================
# 1. Data loading and plotting calculations
# ============================================================
# Figure 1 zebrafish recording-level FC-dynamics table. Panel A and the
# division panels are both rebuilt from this table by z-scoring within each
# recording and then averaging values at the node level for Panel A.
fig1_rec_all = pd.read_csv(os.path.join(DATA, FIG1_RECORDING_MEASURE_TABLE))
fig1_rec_all = fig1_rec_all.loc[fig1_rec_all["species"].eq("Zebrafish")].replace(
    [np.inf, -np.inf], np.nan
).copy()
fig1_rec = fig1_rec_all.copy()
if ARGS.regions_file:
    region_table = pd.read_csv(ARGS.regions_file)
    if "node" not in region_table.columns:
        raise KeyError(f"Region filter lacks a node column: {ARGS.regions_file}")
    selected_nodes = set(region_table["node"].astype(str))
    missing_nodes = selected_nodes.difference(set(fig1_rec["node"].astype(str)))
    if missing_nodes:
        raise ValueError(f"Figure 9 input lacks selected nodes: {sorted(missing_nodes)}")
    fig1_rec = fig1_rec.loc[fig1_rec["node"].astype(str).isin(selected_nodes)].copy()
    print(f"Figure 9 region filter: {fig1_rec['node'].nunique()} nodes, {len(fig1_rec)} rows")
fig1_rec = fig1_rec.rename(columns={"NetTE": "ObservedNetTE"})
fig1_rec_all = fig1_rec_all.rename(columns={"NetTE": "ObservedNetTE"})
required_cols = ["species", "recording_id", "node"] + MEASURE_COLUMNS
missing_cols = [col for col in required_cols if col not in fig1_rec.columns]
if missing_cols:
    raise KeyError(f"Missing required Figure 9 input columns: {missing_cols}")
missing_all_cols = [col for col in required_cols if col not in fig1_rec_all.columns]
if missing_all_cols:
    raise KeyError(f"Missing required unfiltered Figure 9 input columns: {missing_all_cols}")

fig1_zf_rec = fig1_rec[required_cols].copy()
fig1_zf_raw_rec = fig1_zf_rec.copy()
fig1_zf_raw_rec["anatomy_group"] = fig1_zf_raw_rec["node"].map(anatomy_group)
fig1_zf_raw_rec["_group_order"] = fig1_zf_raw_rec["anatomy_group"].map(
    {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
).fillna(len(ZF_GROUP_ORDER)).astype(int)
fig1_zf_raw_rec = fig1_zf_raw_rec.sort_values(
    ["_group_order", "anatomy_group", "recording_id", "node"]
).reset_index(drop=True)

# Unfiltered recording-region table used only for Panels B-D, allowing every
# available region in each subject to contribute a point.
fig1_zf_all_raw_rec = fig1_rec_all[required_cols].copy()
fig1_zf_all_raw_rec["anatomy_group"] = fig1_zf_all_raw_rec["node"].map(anatomy_group)
fig1_zf_all_raw_rec["_group_order"] = fig1_zf_all_raw_rec["anatomy_group"].map(
    {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
).fillna(len(ZF_GROUP_ORDER)).astype(int)
fig1_zf_all_raw_rec = fig1_zf_all_raw_rec.sort_values(
    ["_group_order", "anatomy_group", "recording_id", "node"]
).reset_index(drop=True)

# Panel A: average raw recording-node values at the node level first, then
# z-score each measure across nodes (average-then-zscore).
fig1_zf = (
    fig1_zf_raw_rec.groupby(["node", "anatomy_group"], as_index=False)
    .agg(**{col: (col, "mean") for col in MEASURE_COLUMNS})
    .dropna(subset=MEASURE_COLUMNS, how="any")
)
for _col in MEASURE_COLUMNS:
    fig1_zf[_col] = _zscore(fig1_zf[_col].to_numpy(float))
fig1_zf["_group_order"] = fig1_zf["anatomy_group"].map(
    {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
).fillna(len(ZF_GROUP_ORDER)).astype(int)
fig1_zf = fig1_zf.sort_values(["_group_order", "anatomy_group", "node"]).reset_index(drop=True)

fig1_zf_raw = (
    fig1_zf_raw_rec.groupby(["node", "anatomy_group"], as_index=False)
    .agg(**{col: (col, "mean") for col in MEASURE_COLUMNS})
)
fig1_zf_raw["_group_order"] = fig1_zf_raw["anatomy_group"].map(
    {group: idx for idx, group in enumerate(ZF_GROUP_ORDER)}
).fillna(len(ZF_GROUP_ORDER)).astype(int)
fig1_zf_raw = fig1_zf_raw.sort_values(["_group_order", "anatomy_group", "node"]).reset_index(drop=True)

regions_d = fig1_zf["node"].astype(str).to_numpy()
divs_d = fig1_zf["anatomy_group"].astype(str).to_numpy()
out_data = fig1_zf[[col for _, col in PANEL_A_FEATURES]].to_numpy(float).T

div_color_map = ZF_GROUP_COLORS

# Panel A preserves Ward clustering of the measured FC features, but flips
# dendrogram branches so clusters enriched for Tel tend to appear on the left
# and Hind-enriched clusters on the right when this is possible within the
# existing hierarchy.
Z = linkage(out_data.T, method='ward')
Z = _reorder_linkage_tel_left_hind_right(Z, divs_d)

dend_info = dendrogram(Z, no_plot=True)
leaf_order = np.array(dend_info['leaves'])
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
_output_w = fs.MAIN_FIGURE_WIDTH
_fig_w = 8.4
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

ax_dendro = fig.add_subplot(gs[0, 0:10])
ax_divbar_a = fig.add_subplot(gs[1, 0:10])
ax_heat = fig.add_subplot(gs[2, 0:10])
ax_cbar = fig.add_subplot(gs[2, 11:12])

ax_c = fig.add_subplot(gs[3:5, 0:4])
ax_d = fig.add_subplot(gs[3:5, 4:8])
ax_e = fig.add_subplot(gs[3:5, 8:12])

division_colors = ZF_GROUP_COLORS
division_order = [group for group in ZF_GROUP_ORDER if group in set(fig1_zf["anatomy_group"])]
division_short_labels = {
    "Tel": "Tel",
    "Di": "Di",
    "Mes": "Mes",
    "Hind": "Hind",
}
_STAR_FS = fs.STAR_FS_2COL

def _record_division_stats(panel, df, order):
    groups = [df[k].dropna().values for k in order]
    pairs = list(itertools.combinations(range(len(order)), 2))
    pvals = [
        mannwhitneyu(groups[i], groups[j], alternative='two-sided')[1]
        for i, j in pairs
    ]
    reject, corr, _, _ = multipletests(pvals, method='holm')
    for idx, (i, j) in enumerate(pairs):
        STATS_ROWS.append({
            'figure': 'figure9',
            'panel': panel,
            'test': 'Mann-Whitney U',
            'alternative': 'two-sided',
            'group_1': order[i],
            'group_2': order[j],
            'n_group_1': len(groups[i]),
            'n_group_2': len(groups[j]),
            'p_uncorrected': pvals[idx],
            'p_holm': corr[idx],
            'reject_holm_0.05': bool(reject[idx]),
        })
    return pairs, reject, corr

def _add_sig_bars(ax, df, order, panel):
    pairs, reject, corr = _record_division_stats(panel, df, order)
    sig = sorted([(pairs[k], corr[k]) for k in range(len(pairs)) if reject[k]],
                 key=lambda x: x[0][1] - x[0][0])
    if not sig:
        return
    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    step = yr * 0.090
    bar_h = yr * 0.018
    for lvl, ((i, j), p) in enumerate(sig):
        y = y_max + yr * 0.025 + lvl * step
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        ax.plot(
            [i, i, j, j],
            [y, y + bar_h, y + bar_h, y],
            lw=0.65,
            c='#333',
            clip_on=False,
        )
        ax.text((i + j) / 2, y + bar_h, star,
                ha='center', va='center', fontsize=_STAR_FS,
                clip_on=False)
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
    ax.set_xlabel('')
    ax.set_xticks(range(len(division_order)))
    ax.set_xticklabels(
        [division_short_labels.get(group, group) for group in division_order],
        rotation=0,
        ha='center',
    )
    if ylim is not None:
        ax.set_ylim(*ylim)

def _fig1_measure_division_df(plot_df, measure):
    plot_df = plot_df[['anatomy_group', measure]].dropna().copy()
    grouped = {
        div: plot_df.loc[plot_df['anatomy_group'] == div, measure].to_numpy()
        for div in division_order
    }
    max_len = max((len(v) for v in grouped.values()), default=0)
    return pd.DataFrame({
        div: pd.Series(vals, dtype=float).reindex(range(max_len))
        for div, vals in grouped.items()
    })

# ── Figure 1 zebrafish data (Panels B-D) ──
# Standardize regions within each subject. Each plotted point is one
# subject-region observation; values are not averaged across subjects.
panel_bcd_measures = ['EdgeStdFCV', 'FCS', 'ProfileCorrDistFCV']
fig1_zf_subject_z = fig1_zf_all_raw_rec.copy()
for _col in panel_bcd_measures:
    fig1_zf_subject_z[_col] = fig1_zf_subject_z.groupby('recording_id')[_col].transform(
        lambda values: _zscore(values.to_numpy(dtype=float))
    )

_fcv = _fig1_measure_division_df(fig1_zf_subject_z, 'EdgeStdFCV')
_fcs = _fig1_measure_division_df(fig1_zf_subject_z, 'FCS')
_profile_corr = _fig1_measure_division_df(fig1_zf_subject_z, 'ProfileCorrDistFCV')

# ============================================================
# 3. Draw each panel
# ============================================================
# Panels C-E: division-level boxplots
_boxplot_panel(ax_c, _fcv, 'FCV')
_boxplot_panel(ax_d, _fcs, 'FCS')
_boxplot_panel(ax_e, _profile_corr, 'FC Recon. Deg')



_add_sig_bars(ax_c, {c: _fcv[c] for c in division_order}, division_order, 'FCV')
_add_sig_bars(ax_d, {c: _fcs[c] for c in division_order}, division_order, 'FCS')
_add_sig_bars(ax_e, {c: _profile_corr[c] for c in division_order}, division_order, 'ProfileCorrDistFCV')

# Panel A: dendrogram, division bar, and feature heatmap
dendrogram(Z, ax=ax_dendro, no_labels=True,
           color_threshold=0, above_threshold_color='#333333')
ax_dendro.set_xlim(0, n_regions * 10)
dend_max_h = max(max(d) for d in dend_info['dcoord'])
ax_dendro.set_ylim(0, dend_max_h * 1.05)
ax_dendro.axis('off')

for i, div in enumerate(divs_c):
    ax_divbar_a.add_patch(
        plt.Rectangle(
            (i - 0.5, 0),
            1,
            1,
            color=div_color_map.get(div, 'gray'),
            linewidth=0,
        )
    )

ax_divbar_a.set_xlim(-0.5, n_regions - 0.5)
ax_divbar_a.set_ylim(0, 1)
ax_divbar_a.set_xticks([])
ax_divbar_a.set_yticks([])
for spine in ax_divbar_a.spines.values():
    spine.set_visible(False)

im = ax_heat.imshow(out_data_c, aspect='auto', cmap='RdBu_r',
                    vmax=2, vmin=-2, interpolation='nearest')
ax_heat.set_xlim(-0.5, n_regions - 0.5)

ax_heat.set_yticks(range(len(y_labels)))
ax_heat.set_yticklabels(y_labels, fontsize=fs.TICK_FS_2COL - 2)
ax_heat.tick_params(axis='y', length=0, labelright=False, labelleft=True)

ax_heat.set_xticks(range(n_regions))
ax_heat.set_xticklabels(regions_c, rotation=90, fontsize=fs.TICK_FS_2COL - 2,
                         ha='center')
ax_heat.tick_params(axis='x', length=2, width=0.5)

for tick_label, div in zip(ax_heat.get_xticklabels(), divs_c):
    tick_label.set_color(div_color_map[div])

for y in np.arange(-0.5, len(y_labels), 1):
    ax_heat.axhline(y=y, color='white', linewidth=0.4)

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
    spine.set_color('black')
ax_heat.add_patch(
    mpatches.Rectangle(
        (-0.5, -0.5), n_regions, len(y_labels),
        fill=False, edgecolor='black', linewidth=1.4,
        zorder=10, clip_on=False
    )
)

cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.set_title('Z-score', fontsize=fs.AXIS_LABEL_FS_2COL)
cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)
cbar.set_ticks([-2, -1, 0, 1, 2])

div_handles = [mpatches.Patch(color=div_color_map[d], label=d)
               for d in division_order]
ax_heat.legend(
    handles=div_handles,
    loc='upper left',
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
    _ax.set_position([
        _p.x0 - _p.width * 0.045,
        _p.y0 - _p.height * 0.5,
        _p.width * 1.045,
        _p.height * 1.5,
    ])

for _ax in [ax_dendro, ax_divbar_a]:
    _p = _ax.get_position()
    _ax.set_position([
        _p.x0 - _p.width * 0.045,
        _p.y0,
        _p.width * 1.045,
        _p.height,
    ])

_cb_pos = ax_cbar.get_position()
ax_cbar.set_position([_cb_pos.x0, _cb_pos.y0, _cb_pos.width * 0.4, _cb_pos.height * 1.5])


_cb_pos = ax_cbar.get_position()
ax_cbar.set_position([_cb_pos.x0-0.07, _cb_pos.y0-0.0715, _cb_pos.width, _cb_pos.height])

for _ax in [ax_c, ax_d, ax_e]:
    _p = _ax.get_position()
    _cx = _p.x0 + _p.width / 2
    _cy = _p.y0 + _p.height / 2
    _nw = _p.width * 0.78
    _nh = _p.height * 0.50
    _ax.set_position([_cx - _nw / 2, _cy - _nh / 2, _nw, _nh])

_add_figure9_panel_labels(ax_dendro, ax_c, ax_d, ax_e)


# ============================================================
# 5. Save figure and statistics
# ============================================================
fig.canvas.draw()
_content_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
_output_h = 6.4
_output_bbox = Bbox.from_bounds(
    _content_bbox.x0 - (_output_w - _content_bbox.width) / 2,
    _content_bbox.y0 - (_output_h - _content_bbox.height) / 2,
    _output_w,
    _output_h,
)
fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches=_output_bbox, transparent=False)
#fig.savefig(OUTPUT_PDF, bbox_inches='tight', transparent=False)
os.makedirs(STATS_DIR, exist_ok=True)
pd.DataFrame(STATS_ROWS).to_csv(STATS_CSV, index=False)
print(f"Saved {STATS_CSV}")
