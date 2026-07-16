import importlib.util
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

import figure_style as fig_help

# Workflow:
# 1. Data loading and plotting calculations.
# 2. Layout preparation.
# 3. Draw each panel.
# 4. Panel position adjustment and panel labels.
# 5. Save figure and statistics.

warnings.filterwarnings("ignore", category=FutureWarning)

fig_help.apply_main_figure_style()


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "raw_data" / "figure_supply_15"
FIG13_DATA_DIR = DATA_DIR / "figure13_inputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
STATS_DIR = FIG13_DATA_DIR
STATS_CSV = STATS_DIR / "figure13_stats.csv"
OUT_PNG = FIGURE_DIR / "figure13_final.png"
ENERGY_POTENTIAL_CURVES = STATS_DIR / "figure13_layer_energy_potential_curves.csv"
ENERGY_POTENTIAL_SUMMARY = STATS_DIR / "figure13_layer_energy_potential_summary.csv"
ENERGY_WIDTH_DELTA_U = 2.0

PANEL_FS = fig_help.MAIN_PANEL_LABEL_FS
TITLE_FS = fig_help.MAIN_TITLE_FS
AXIS_FS = fig_help.MAIN_AXIS_FS
TICK_FS = fig_help.MAIN_TICK_FS
LEGEND_FS = fig_help.MAIN_TICK_FS
STAR_FS = fig_help.MAIN_STAR_FS
LINE_W = fig_help.MAIN_LINE_W
MARKER_SIZE = fig_help.MAIN_MARKER_SIZE


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


layer_plot = _load_module(
    BASE_DIR / "plot_layer_asymmetric_epsilon_linear_model.py",
    "figure13_layer_asymmetric_plot",
)


# 1. Data loading and plotting calculations

def bootstrap_diff(a_values, b_values, n_boot=10000):
    observed = np.mean(a_values) - np.mean(b_values)
    diffs = []
    for _ in range(n_boot):
        a_star = np.random.choice(a_values, size=len(a_values), replace=True)
        b_star = np.random.choice(b_values, size=len(b_values), replace=True)
        diffs.append(np.mean(a_star) - np.mean(b_star))
    diffs = np.asarray(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return observed, ci_lower, ci_upper, p_value, diffs


def _sig_star(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def run_three_bootstrap(out_data, in_data, base_data, n_boot=10000, holm_swap=False):
    r1 = bootstrap_diff(out_data, base_data, n_boot=n_boot)
    r2 = bootstrap_diff(in_data, base_data, n_boot=n_boot)
    r3 = bootstrap_diff(out_data, in_data, n_boot=n_boot)
    obs1, ci1_lo, ci1_hi, p1, d1 = r1
    obs2, ci2_lo, ci2_hi, p2, d2 = r2
    obs3, ci3_lo, ci3_hi, p3, d3 = r3
    if holm_swap:
        _, corr, _, _ = multipletests([p2, p1, p3], alpha=0.05, method="holm")
        p_ob_c, p_ib_c, p_oi_c = corr[1], corr[0], corr[2]
    else:
        _, corr, _, _ = multipletests([p1, p2, p3], alpha=0.05, method="holm")
        p_ob_c, p_ib_c, p_oi_c = corr[0], corr[1], corr[2]
    rows = [
        {
            "comparison": "Out-Base",
            "observed_difference": obs1,
            "ci_2.5": ci1_lo,
            "ci_97.5": ci1_hi,
            "p_uncorrected": p1,
            "p_holm": p_ob_c,
        },
        {
            "comparison": "In-Base",
            "observed_difference": obs2,
            "ci_2.5": ci2_lo,
            "ci_97.5": ci2_hi,
            "p_uncorrected": p2,
            "p_holm": p_ib_c,
        },
        {
            "comparison": "Out-In",
            "observed_difference": obs3,
            "ci_2.5": ci3_lo,
            "ci_97.5": ci3_hi,
            "p_uncorrected": p3,
            "p_holm": p_oi_c,
        },
    ]
    return (d1, d2, d3), (p_ob_c, p_ib_c, p_oi_c), rows


def plot_violin_bootstrap(ax, boot_data, p_values, ylabel, colors_violin):
    positions = [0, 1, 2]
    parts = ax.violinplot(
        boot_data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc, color in zip(parts["bodies"], colors_violin):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")

    box_w = 0.10
    for pos, data in zip(positions, boot_data):
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        iqr = q3 - q1
        wlo = max(np.min(data), q1 - 1.5 * iqr)
        whi = min(np.max(data), q3 + 1.5 * iqr)
        ax.plot([pos, pos], [wlo, whi], color="black", linewidth=0.8, zorder=3)
        rect = plt.Rectangle(
            (pos - box_w / 2, q1),
            box_w,
            iqr,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.scatter([pos], [med], color="white", s=12, zorder=5,
                   edgecolors="black", linewidths=0.7)

    y_max = max(np.max(d) for d in boot_data)
    y_min = min(np.min(d) for d in boot_data)
    y_range = y_max - y_min
    y_step = y_range * 0.04
    for pos, star in zip(positions, (_sig_star(p) for p in p_values)):
        ax.text(pos, y_max + y_step, star, ha="center", va="bottom",
                fontsize=STAR_FS)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.14)
    ax.set_xlim(-0.7, 2.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Out-B", "In-B", "Out-In"], ha="right", rotation=35,
                       fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, bottom=True, labelbottom=True, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_figure11_layer_data():
    data_path = FIG13_DATA_DIR / "layer_asymmetric_epsilon_linear_data.npz"
    return layer_plot.load_results(data_path)


def load_figure10_large_scale_bootstrap():
    np.random.seed(0)
    fig5_path = FIG13_DATA_DIR / "tfigure5_compare_xy_scatter_data.pkl"
    figure1_path = FIG13_DATA_DIR / "figure1_whole_brain_fc_mean_std_data.npz"
    figure6_dir = FIG13_DATA_DIR

    with open(fig5_path, "rb") as pf:
        fig5 = pickle.load(pf)
    with open(figure6_dir / "wsim_fc_null_p_sp_out_fc_mean_std_data.pkl", "rb") as pf:
        fig6_in = pickle.load(pf)
    with open(figure6_dir / "wsim_fc_null_p_sp_in_fc_mean_std_data.pkl", "rb") as pf:
        fig6_out = pickle.load(pf)

    emp_regions = np.load(figure1_path)["new_region_array"]

    sim_ave_data = []
    in_sim_ave_data = []
    out_sim_ave_data = []
    sim_std_data = []
    in_sim_std_data = []
    out_sim_std_data = []

    selected_regions = {22, 28, 22 + 36, 28 + 36}
    for region_idx in emp_regions:
        region_idx = int(region_idx)
        if region_idx in selected_regions:
            sim_ave_data.extend(fig5["sim_ave_fc_list"][region_idx])
            in_sim_ave_data.extend(fig6_in["sim_ave_fc_list"][region_idx])
            out_sim_ave_data.extend(fig6_out["sim_ave_fc_list"][region_idx])
            sim_std_data.extend(fig5["sim_std_fc_list"][region_idx])
            in_sim_std_data.extend(fig6_in["sim_std_fc_list"][region_idx])
            out_sim_std_data.extend(fig6_out["sim_std_fc_list"][region_idx])

    stats_rows = []
    fcs_kruskal = kruskal(out_sim_ave_data, in_sim_ave_data, sim_ave_data)
    (d1, d2, d3), (p1, p2, p3), fcs_rows = run_three_bootstrap(
        out_sim_ave_data, in_sim_ave_data, sim_ave_data, n_boot=10000, holm_swap=False
    )
    fcs_boot = [d1, d2, d3]
    fcs_p = [p1, p2, p3]
    stats_rows.append({
        "figure": "figure13",
        "panel": "E",
        "metric": "FCS",
        "test": "Kruskal-Wallis",
        "comparison": "Base vs Null-In vs Null-Out",
        "statistic": fcs_kruskal.statistic,
        "p_value": fcs_kruskal.pvalue,
        "n_base": len(sim_ave_data),
        "n_null_in": len(in_sim_ave_data),
        "n_null_out": len(out_sim_ave_data),
        "mean_base": np.mean(sim_ave_data),
        "mean_null_in": np.mean(in_sim_ave_data),
        "mean_null_out": np.mean(out_sim_ave_data),
    })
    for row in fcs_rows:
        row.update({
            "figure": "figure13",
            "panel": "E",
            "metric": "FCS",
            "test": "bootstrap mean difference",
        })
        stats_rows.append(row)

    fcv_kruskal = kruskal(out_sim_std_data, in_sim_std_data, sim_std_data)
    (d4, d5, d6), (p4, p5, p6), fcv_rows = run_three_bootstrap(
        out_sim_std_data, in_sim_std_data, sim_std_data, n_boot=10000, holm_swap=False
    )
    fcv_boot = [d4, d5, d6]
    fcv_p = [p4, p5, p6]
    stats_rows.append({
        "figure": "figure13",
        "panel": "E",
        "metric": "FCV",
        "test": "Kruskal-Wallis",
        "comparison": "Base vs Null-In vs Null-Out",
        "statistic": fcv_kruskal.statistic,
        "p_value": fcv_kruskal.pvalue,
        "n_base": len(sim_std_data),
        "n_null_in": len(in_sim_std_data),
        "n_null_out": len(out_sim_std_data),
        "mean_base": np.mean(sim_std_data),
        "mean_null_in": np.mean(in_sim_std_data),
        "mean_null_out": np.mean(out_sim_std_data),
    })
    for row in fcv_rows:
        row.update({
            "figure": "figure13",
            "panel": "E",
            "metric": "FCV",
            "test": "bootstrap mean difference",
        })
        stats_rows.append(row)
    return (fcs_boot, fcs_p), (fcv_boot, fcv_p), stats_rows


def prepare_plot_data():
    results, args = load_figure11_layer_data()
    eps = results["epsilon_values"]
    mean_fc_mean, mean_fc_sem = layer_plot.mean_and_sem(results["layer_mean_fc"])
    std_mean, std_sem = layer_plot.mean_and_sem(results["layer_temporal_std_fc"])
    (fcs_boot, fcs_p), (fcv_boot, fcv_p), stats_rows = load_figure10_large_scale_bootstrap()
    return {
        "args": args,
        "eps": eps,
        "mean_fc_mean": mean_fc_mean,
        "mean_fc_sem": mean_fc_sem,
        "std_mean": std_mean,
        "std_sem": std_sem,
        "fcs_boot": fcs_boot,
        "fcs_p": fcs_p,
        "fcv_boot": fcv_boot,
        "fcv_p": fcv_p,
        "stats_rows": stats_rows,
    }


def draw_layer_fc_panel(ax, eps, values, sem, args, ylabel, title, show_legend=False):
    layer_colors = [
        fig_help.ZEBRAFISH_DIVISION_COLORS["Tel"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Di"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Mes"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Hind"],
    ]
    for layer_idx in range(len(args.layer_sizes)):
        label = f"layer {layer_idx + 1}"
        color = layer_colors[layer_idx % len(layer_colors)]
        ax.fill_between(
            eps,
            values[:, layer_idx] - sem[:, layer_idx],
            values[:, layer_idx] + sem[:, layer_idx],
            color=color,
            alpha=0.18,
        )
        ax.plot(
            eps,
            values[:, layer_idx],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_W,
            color=color,
            label=label,
        )
    ax.set_xlabel(r"$\epsilon$", fontsize=AXIS_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.set_title("")
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=1.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=LEGEND_FS,
            handlelength=1.2,
            labelspacing=0.25,
            borderaxespad=0.2,
            markerscale=0.7,
        )


def _mean_sem_nan(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean = np.full(values.shape[1], np.nan)
    sem = np.full(values.shape[1], np.nan)
    ok = count > 0
    mean[ok] = np.nansum(values[:, ok], axis=0) / count[ok]
    if np.any(count > 1):
        sem[count > 1] = np.nanstd(values[:, count > 1], axis=0, ddof=1) / np.sqrt(count[count > 1])
    return mean, sem


def _well_width(z, potential, delta_u=ENERGY_WIDTH_DELTA_U):
    """Contiguous z-range around the minimum with U(z) <= U_min + delta_u."""
    z = np.asarray(z, dtype=float)
    potential = np.asarray(potential, dtype=float)
    finite = np.isfinite(z) & np.isfinite(potential)
    z, potential = z[finite], potential[finite]
    if len(z) < 3:
        return np.nan
    order = np.argsort(z)
    z, potential = z[order], potential[order]
    minimum = int(np.argmin(potential))
    threshold = potential[minimum] + delta_u
    lower = minimum
    while lower > 0 and potential[lower - 1] <= threshold:
        lower -= 1
    upper = minimum
    while upper < len(potential) - 1 and potential[upper + 1] <= threshold:
        upper += 1
    return float(z[upper] - z[lower])


def load_energy_width(delta_u=ENERGY_WIDTH_DELTA_U):
    """Calculate per-run energy-well widths from the curves shown in panel A."""
    if not ENERGY_POTENTIAL_CURVES.exists():
        raise FileNotFoundError(ENERGY_POTENTIAL_CURVES)
    curves = pd.read_csv(ENERGY_POTENTIAL_CURVES)
    rows = []
    for (epsilon, layer, run), group in curves.groupby(["epsilon", "layer", "run"]):
        rows.append(
            {
                "epsilon": float(epsilon),
                "layer": str(layer),
                "run": int(run),
                "energy_well_width": _well_width(
                    group["dynamic_fc_state"].to_numpy(float),
                    group["potential"].to_numpy(float),
                    delta_u,
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_energy_width_panel(ax, width_df):
    """Plot mean energy-well width across a dense epsilon grid for all layers."""
    colors = {
        "Layer 1": fig_help.ZEBRAFISH_DIVISION_COLORS["Tel"],
        "Layer 2": fig_help.ZEBRAFISH_DIVISION_COLORS["Di"],
        "Layer 3": fig_help.ZEBRAFISH_DIVISION_COLORS["Mes"],
        "Layer 4": fig_help.ZEBRAFISH_DIVISION_COLORS["Hind"],
    }
    for layer in ["Layer 1", "Layer 2", "Layer 3", "Layer 4"]:
        group = width_df.loc[width_df["layer"].eq(layer)].dropna(
            subset=["energy_well_width"]
        )
        summary = (
            group.groupby("epsilon")["energy_well_width"]
            .agg(["mean", "sem"])
            .sort_index()
        )
        x = summary.index.to_numpy(float)
        mean = summary["mean"].to_numpy(float)
        sem = summary["sem"].to_numpy(float)
        color = colors[layer]
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.12,
                        linewidth=0)
        ax.errorbar(
            x, mean, yerr=sem, marker="o", markersize=MARKER_SIZE * 0.75,
            linewidth=LINE_W * (1.35 if layer == "Layer 1" else 1.0),
            elinewidth=0.65, capsize=1.2,
            color=color, label=layer,
        )
    ax.set_xlabel(r"$\epsilon$", fontsize=AXIS_FS)
    ax.set_ylabel("Energy well width", fontsize=AXIS_FS)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.08, 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_FS - 1,
              handlelength=1.0, borderaxespad=0.2, ncol=2,
              columnspacing=0.7, handletextpad=0.35)


def draw_energy_potential_panel(ax):
    if not ENERGY_POTENTIAL_CURVES.exists() or not ENERGY_POTENTIAL_SUMMARY.exists():
        raise FileNotFoundError(
            "Energy potential CSVs not found. Run figure13_layer_energy_potential.py first."
        )
    curves = pd.read_csv(ENERGY_POTENTIAL_CURVES)
    summary = pd.read_csv(ENERGY_POTENTIAL_SUMMARY)
    ax.set_axis_off()

    eps_values = [0.0, 0.5, 1.0]
    eps_colors = {
        0.0: fig_help.MAIN_COLORS["neutral_dark"],
        0.5: fig_help.MAIN_COLORS["post_out"],
        1.0: fig_help.MAIN_COLORS["pre_in"],
    }
    
    layer_axes = [
        ax.inset_axes([0.0, 0.60, 1.0, 0.4]),
        ax.inset_axes([0.0, 0.00, 1.0, 0.4]),
    ]
    
    
    layers = ["Layer 1", "Layer 4"]

    for layer_ax, layer_name in zip(layer_axes, layers):
        layer_ax.spines[["top", "right"]].set_visible(False)
        layer_ax.grid(False)
        for eps in eps_values:
            sub = curves[
                curves["layer"].eq(layer_name)
                & np.isclose(curves["epsilon"].astype(float), eps)
            ]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index="run",
                columns="dynamic_fc_state",
                values="potential",
                aggfunc="mean",
            ).sort_index(axis=1)
            x = pivot.columns.to_numpy(dtype=float)
            y_mean, y_sem = _mean_sem_nan(pivot.to_numpy(dtype=float))
            mask = np.isfinite(y_mean)
            y = np.clip(y_mean[mask], 0, 12)
            err = np.nan_to_num(y_sem[mask], nan=0.0)
            color = eps_colors[eps]
            layer_ax.plot(x[mask], y, color=color, lw=1.25, label=rf"$\epsilon={eps:g}$")
            layer_ax.fill_between(
                x[mask],
                np.clip(y - err, 0, 12),
                np.clip(y + err, 0, 12),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        fc_rows = summary[summary["layer"].eq(layer_name)]
        text_lines = []
        for eps in eps_values:
            vals = fc_rows.loc[
                np.isclose(fc_rows["epsilon"].astype(float), eps),
                "fcv_from_fc_state",
            ].dropna()
            if len(vals):
                text_lines.append(rf"$\epsilon={eps:g}$: {vals.mean():.2f}$\pm${vals.sem():.2f}")
        '''
        if text_lines:
            layer_ax.text(
                0.02,
                0.95,
                "FCV\n" + "\n".join(text_lines),
                transform=layer_ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.8,
                bbox=dict(facecolor="white", edgecolor="0.85", lw=0.45, alpha=0.78, pad=1.8),
            )
        '''
        layer_ax.set_title(layer_name, fontsize=AXIS_FS, pad=2.5, fontweight="bold")
        layer_ax.set_ylabel(r"$U(z)$", fontsize=AXIS_FS)
        layer_ax.set_ylim(0, 12)
        layer_ax.set_yticks([0, 5, 10])
        layer_ax.tick_params(axis="both", labelsize=AXIS_FS, pad=1)
        #layer_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        #layer_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
    layer_axes[0].set_xlim(-1.0,1.5)   
    layer_axes[1].set_xlim(-1.0,1.5)   
    #layer_axes[0].set_xticklabels([])
    layer_axes[1].set_xlabel("dynamic FC state", fontsize=AXIS_FS, labelpad=1)
    layer_axes[0].legend(
        loc="lower left",
        frameon=False,
        fontsize=7.9,
        handlelength=1.2,
        borderaxespad=0.1,
    )
    layer_axes[1].legend(
        loc="lower left",
        frameon=False,
        fontsize=7.9,
        handlelength=1.2,
        borderaxespad=0.1,
    )

def draw_layer_network_panel_clean(ax, args):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    layer_sizes = list(args.layer_sizes)
    n_layers = len(layer_sizes)
    y_positions = np.linspace(0.96, 0.06, n_layers)
    x_center = 0.44
    node_spacing = 0.135
    max_size = max(layer_sizes)
    positions = []
    layer_colors = [
        fig_help.ZEBRAFISH_DIVISION_COLORS["Tel"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Di"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Mes"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Hind"],
    ]

    for layer_idx, size in enumerate(layer_sizes):
        xs = x_center + (np.arange(size) - (size - 1) / 2.0) * node_spacing
        ys = np.full(size, y_positions[layer_idx])
        positions.append(np.column_stack([xs, ys]))

    for layer_idx, layer_xy in enumerate(positions):
        layer_color = layer_colors[layer_idx % len(layer_colors)]
        face_color = mcolors.to_rgba(layer_color, 0.68)
        edge_color = mcolors.to_rgba(layer_color, 0.95)
        ax.scatter(
            layer_xy[:, 0],
            layer_xy[:, 1],
            s=145,
            facecolors=[face_color],
            edgecolors=[edge_color],
            linewidth=1.25,
            zorder=3,
        )
        ax.text(
            0.095-0.0,
            y_positions[layer_idx],
            f"L{layer_idx + 1}\nn={layer_sizes[layer_idx]}",
            ha="right",
            va="center",
            fontsize=TICK_FS,
            linespacing=0.95,
        )

    max_scale = max(args.inter_epsilon_scales)
    for layer_idx in range(n_layers - 1):
        scale = args.inter_epsilon_scales[layer_idx]
        upper = positions[layer_idx]
        lower = positions[layer_idx + 1]
        for src_xy in upper:
            for dst_xy in lower:
                src = src_xy + np.array([0.006, -0.010])
                dst = dst_xy + np.array([0.006, 0.010])
                ax.add_patch(FancyArrowPatch(
                    dst - np.array([0.012, 0.0]),
                    src - np.array([0.012, 0.0]),
                    arrowstyle="-|>",
                    mutation_scale=4.8,
                    lw=1.68,
                    color=fig_help.MAIN_COLORS["pre_in"],
                    alpha=0.35,
                    shrinkA=6,
                    shrinkB=6,
                    connectionstyle="arc3,rad=0.06",
                    zorder=0,
                ))
                ax.add_patch(FancyArrowPatch(
                    src,
                    dst,
                    arrowstyle="-|>",
                    mutation_scale=5.8,
                    lw=1.68,#0.46 + 0.42 * scale / max_scale,
                    color=fig_help.MAIN_COLORS["post_out"],
                    alpha=0.35,#0.135 + 0.115 * scale / max_scale,
                    shrinkA=6,
                    shrinkB=6,
                    connectionstyle="arc3,rad=-0.06",
                    zorder=1,
                ))

    arrow_x = 0.82
    for layer_idx in range(n_layers - 1):
        scale = args.inter_epsilon_scales[layer_idx]
        y0 = y_positions[layer_idx]
        y1 = y_positions[layer_idx + 1]
        lw_down = 2.5 + 3.4 * scale / max_scale
        lw_up = 2.5 + 3.4 * (1 - scale / max_scale)

        ax.add_patch(FancyArrowPatch(
            (arrow_x, y0 - 0.01),
            (arrow_x, y1 + 0.01),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=lw_down,
            color=fig_help.MAIN_COLORS["post_out"],
            alpha=0.92,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.24",
            zorder=4,
        ))
        ax.add_patch(FancyArrowPatch(
            (arrow_x + 0.095, y1 + 0.01),
            (arrow_x + 0.095, y0 - 0.01),
            arrowstyle="-|>",
            mutation_scale=15,
            lw=lw_up,
            color=fig_help.MAIN_COLORS["pre_in"],
            alpha=0.82,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.24",
            zorder=4,
        ))

    ax.text(
        0.75,
        -0.055,
        r"down: $w(1+\epsilon)$",
        ha="center",
        va="center",
        fontsize=TICK_FS ,
        color=fig_help.MAIN_COLORS["post_out"],
        fontweight="bold",
    )
    ax.text(
        0.75,
        -0.135,
        r"up: $w(1-\epsilon)$",
        ha="center",
        va="center",
        fontsize=TICK_FS ,
        color=fig_help.MAIN_COLORS["pre_in"],
        fontweight="bold",
    )


def draw_large_scale_model_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    orange = fig_help.MAIN_COLORS["post_out"]
    blue = fig_help.MAIN_COLORS["pre_in"]

    def _draw_region(ax, cx, cy, w, h, face, edge, label=None, alpha=1.0):
        patch = mpatches.Ellipse(
            (cx, cy),
            w,
            h,
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)
        if label:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=TICK_FS - 4, color=edge, fontweight="bold", zorder=5)

    def _draw_cells(ax, xy, color="white", edge="0.25", alpha=1.0):
        for px, py in xy:
            ax.scatter(
                [px],
                [py],
                s=15,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.45,
                alpha=alpha,
                zorder=4,
            )

    def _arrow(ax, start, end, color, lw=1.0, alpha=1.0, rad=0.0, linestyle="-", zorder=3,
               mutation_scale=7.5):
        ax.add_patch(FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=color,
            alpha=alpha,
            linestyle=linestyle,
            shrinkA=2.0,
            shrinkB=2.0,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        ))

    def _edge_set(ax, starts, ends, color, active=True, rewired=False, reverse=False):
        base_pairs = [(0, 1), (1, 3), (2, 0), (3, 2)]
        rewired_pairs = [(0, 3), (1, 2), (2, 1), (3, 0), (0, 2), (3, 1)]
        pairs = base_pairs
        if rewired:
            pairs = rewired_pairs
        for pair_idx, (start_idx, end_idx) in enumerate(pairs):
            p0 = starts[start_idx]
            p1 = ends[end_idx]
            if reverse:
                p0, p1 = p1, p0
            rad = 0.20 * (-1 if pair_idx % 2 else 1) if rewired else 0.0
            _arrow(
                ax,
                p0,
                p1,
                color,
                lw=1.35, #if (active and rewired) else (1.15 if active else 0.55),
                alpha=0.94, #if (active and rewired) else (0.88 if active else 0.18),
                rad=rad,
                linestyle="--" if rewired else "-",
                zorder=5 if (active and rewired) else (3 if active else 2),
                mutation_scale=9.5 if (active and rewired) else (8.0 if active else 5.0),
            )

    def _draw_condition(x0, title, active):
        if active == "out":
            face = "#F5E4D0"
        elif active == "in":
            face = "#DDE7F0"
        else:
            face = fig_help.MAIN_COLORS["neutral_fill"]
        box = mpatches.FancyBboxPatch(
            (x0, 0.06),
            0.28,
            0.88,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=face,
            edgecolor="0.65",
            linewidth=0.65,
            alpha=0.50,
            zorder=0,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.14, 0.915, title, ha="center", va="center",
                fontsize=TICK_FS - 2, fontweight="bold")

        selected = (x0 + 0.14, 0.51)
        source = (x0 + 0.14, 0.81)
        target = (x0 + 0.14, 0.20)
        cell_dx, cell_dy = 0.040, 0.030
        source_cells = np.array([
            [source[0] - cell_dx, source[1] + cell_dy],
            [source[0] + cell_dx, source[1] + cell_dy],
            [source[0] - cell_dx, source[1] - cell_dy],
            [source[0] + cell_dx, source[1] - cell_dy],
        ])
        selected_cells = np.array([
            [selected[0] - cell_dx, selected[1] + cell_dy],
            [selected[0] + cell_dx, selected[1] + cell_dy],
            [selected[0] - cell_dx, selected[1] - cell_dy],
            [selected[0] + cell_dx, selected[1] - cell_dy],
        ])
        target_cells = np.array([
            [target[0] - cell_dx, target[1] + cell_dy],
            [target[0] + cell_dx, target[1] + cell_dy],
            [target[0] - cell_dx, target[1] - cell_dy],
            [target[0] + cell_dx, target[1] - cell_dy],
        ])

        _draw_region(ax, *source, 0.130, 0.110, fig_help.MAIN_COLORS["neutral_fill"], "0.40", None, alpha=0.70)
        _draw_region(ax, *target, 0.130, 0.110, fig_help.MAIN_COLORS["neutral_fill"], "0.40", None, alpha=0.70)
        _draw_region(ax, *selected, 0.145, 0.125, "#E3F0E1", "#2F6B35", None, alpha=0.95)
        ax.text(
            selected[0] + 0.080,
            selected[1],
            "P/SP",
            ha="left",
            va="center",
            fontsize=TICK_FS - 1,
            color="#2F6B35",
            fontweight="bold",
            zorder=7,
        )

        _edge_set(ax, source_cells, selected_cells, blue,
                  active=active in {"base", "in"}, rewired=active == "in")
        _edge_set(ax, selected_cells, target_cells, orange,
                  active=active in {"base", "out"}, rewired=active == "out")
        _draw_cells(ax, source_cells, color="#DDE7F0", edge=blue, alpha=0.98)
        _draw_cells(ax, target_cells, color="#F5E4D0", edge=orange, alpha=0.98)
        _draw_cells(ax, selected_cells, color="white", edge="#2F6B35", alpha=0.95)

    _draw_condition(0.03, "Base", "base")
    _draw_condition(0.36, "NULL-Out", "out")
    _draw_condition(0.69, "NULL-In", "in")


# 2. Layout preparation

def prepare_layout():
    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(
        1,
        5,
        figure=fig,
        left=0.055,
        right=0.990,
        top=0.80,
        bottom=0.20,
        wspace=0.40,
        width_ratios=[1.12, 1.35, 1.15, 1.18, 0.88],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e_fcv = fig.add_subplot(gs[0, 4])
    axes = {
        "a": ax_a,
        "b": ax_b,
        "c": ax_c,
        "d": ax_d,
        "e_fcv": ax_e_fcv,
    }
    return fig, axes


# 3. Draw each panel

def draw_panel_a(ax_a, plot_data):
    draw_layer_network_panel_clean(ax_a, plot_data["args"])


def draw_panel_b(ax_b, plot_data):
    draw_energy_potential_panel(ax_b)


def draw_panel_c(ax_c, plot_data):
    draw_layer_fc_panel(
        ax_c,
        plot_data["eps"],
        plot_data["std_mean"],
        plot_data["std_sem"],
        plot_data["args"],
        "FCV",
        "Layer-mean std FC",
        show_legend=True,
    )


def draw_panel_d(ax_d):
    draw_large_scale_model_panel(ax_d)


def draw_panel_e(ax_e_fcv, plot_data):
    large_scale_colors = [
        fig_help.MAIN_COLORS["post_out"],
        fig_help.MAIN_COLORS["pre_in"],
        fig_help.MAIN_COLORS["neutral_light"],
    ]
    plot_violin_bootstrap(
        ax_e_fcv,
        plot_data["fcv_boot"],
        plot_data["fcv_p"],
        r"$\Delta$FCV",
        large_scale_colors,
    )
    ax_e_fcv.set_title("")
    ax_e_fcv.yaxis.label.set_color(fig_help.MAIN_COLORS["neutral_dark"])


def draw_all_panels(axes, plot_data):
    draw_panel_a(axes["a"], plot_data)
    draw_panel_b(axes["b"], plot_data)
    draw_panel_c(axes["c"], plot_data)
    draw_panel_d(axes["d"])
    draw_panel_e(axes["e_fcv"], plot_data)


# 4. Panel position adjustment and panel labels

def adjust_panel_positions_and_labels(fig, axes):
    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]
    ax_e_fcv = axes["e_fcv"]

    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0 + 0.012, pos_b.y0, pos_b.width, pos_b.height])
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0 + 0.015, pos_c.y0, pos_c.width, pos_c.height])
    pos_d = ax_d.get_position()
    ax_d.set_position([
        pos_d.x0 - 0.014,
        pos_d.y0 - 0.045,
        pos_d.width,
        pos_d.height + 0.090,
    ])

    fig.canvas.draw()
    linear_left = ax_a.get_position().x0
    linear_right = ax_c.get_position().x1
    large_left = ax_d.get_position().x0
    large_right = ax_e_fcv.get_position().x1
    header_y = max(ax_a.get_position().y1, ax_b.get_position().y1, ax_c.get_position().y1) + 0.105
    fig.text(
        (linear_left + linear_right) / 2,
        header_y,
        "Layer linear model",
        ha="center",
        va="bottom",
        fontsize=TITLE_FS,
        fontweight="bold",
    )
    fig.text(
        (large_left + large_right) / 2,
        header_y,
        "Zebrafish whole-brain network model",
        ha="center",
        va="bottom",
        fontsize=TITLE_FS,
        fontweight="bold",
    )

    label_positions = {
        "A": (-0.16, 1.04),
        "B": (-0.23, 1.04),
        "C": (-0.30, 1.04),
        "D": (-0.12, 0.98),
        "E": (-0.45, 1.04),
    }

    for ax, label in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (ax_e_fcv, "E")]:
        x, y = label_positions[label]
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_FS,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


# 5. Save figure and statistics

def save_figure(fig):
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=False)


def save_statistics(plot_data):
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_data["stats_rows"]).to_csv(STATS_CSV, index=False)
    print(f"Saved {STATS_CSV}")


def main():
    plot_data = prepare_plot_data()
    fig, axes = prepare_layout()
    draw_all_panels(axes, plot_data)
    adjust_panel_positions_and_labels(fig, axes)
    save_figure(fig)
    save_statistics(plot_data)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
