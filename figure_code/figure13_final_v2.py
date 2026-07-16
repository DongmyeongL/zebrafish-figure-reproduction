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
from scipy.stats import kruskal, mannwhitneyu, ttest_1samp, wilcoxon
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
# Match the typeface used in figure9_clean_v2.py (Nimbus Sans, a Helvetica clone).
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


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DERIVED_DIR = PROJECT_ROOT / "derived_data" / "figure13"
RAW_DIR = PROJECT_ROOT / "raw_data" / "figure13"
MODEL_INPUT_DIR = (
    PROJECT_ROOT / "raw_data" / "figure_supply_15" / "figure13_inputs"
)
FIGURE_DIR = PROJECT_ROOT / "figures"
STATS_DIR = PROJECT_ROOT / "statistics"
STATS_CSV = STATS_DIR / "figure13_stats_v2.csv"
OUT_PNG = FIGURE_DIR / "figure13_final_v2.png"
ENERGY_POTENTIAL_CURVES = MODEL_INPUT_DIR / "figure13_layer_energy_potential_curves.csv"
ENERGY_POTENTIAL_SUMMARY = MODEL_INPUT_DIR / "figure13_layer_energy_potential_summary.csv"
# All-layer, dense-epsilon recompute (figure13_layer_energy_potential_v1.py).
ENERGY_POTENTIAL_CURVES_V1 = DERIVED_DIR / "figure13_layer_energy_potential_curves_v1.csv"
# Per-(epsilon, run, layer) summary from the SAME 50-run v1 simulation; holds
# fcv_from_fc_state, letting panel-B FCV share the width source at n=50.
ENERGY_POTENTIAL_SUMMARY_V1 = DERIVED_DIR / "layer_fcv_dense_summary.csv"
ENERGY_WIDTH_DENSE = DERIVED_DIR / "layer_energy_width_dense.csv"

# Well width is measured as the z-range where U(z) <= U_min + ENERGY_WIDTH_DELTA_U.
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
        ax.text(pos, 0.1, star, ha="center", va="bottom",
                fontsize=STAR_FS)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.14)
    ax.set_xlim(-0.7, 2.7)
    ax.set_xticks(positions)
    #ax.set_ylim(-0.85,0.15);
    ax.set_xticklabels(["NULLOut-Base", "NULLIn-Base", "NULLOut-NULLIn"], ha="center", rotation=30,
                       fontsize=TICK_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, bottom=True, labelbottom=True, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_figure11_layer_data():
    data_path = MODEL_INPUT_DIR / "layer_asymmetric_epsilon_linear_data.npz"
    return layer_plot.load_results(data_path)


def _well_width(z, U, dU):
    """Width of the potential well: contiguous z-range around U_min with U <= U_min + dU."""
    z = np.asarray(z, dtype=float)
    U = np.asarray(U, dtype=float)
    order = np.argsort(z)
    z, U = z[order], U[order]
    finite = np.isfinite(U) & np.isfinite(z)
    z, U = z[finite], U[finite]
    if len(z) < 3:
        return np.nan
    imin = int(np.argmin(U))
    thr = U[imin] + dU
    lo = imin
    while lo > 0 and U[lo - 1] <= thr:
        lo -= 1
    hi = imin
    while hi < len(U) - 1 and U[hi + 1] <= thr:
        hi += 1
    return z[hi] - z[lo]


def load_energy_width(dU=ENERGY_WIDTH_DELTA_U):
    """Per-run energy well width from the U(z) curves, plus panel-B summary stats rows.

    Uses the all-layer, dense-epsilon recompute (ENERGY_POTENTIAL_CURVES_V1) when
    present, otherwise falls back to the base Layer 1 / Layer 4 curves.
    """
    if ENERGY_WIDTH_DENSE.exists():
        dense = pd.read_csv(ENERGY_WIDTH_DENSE)
        return dense.rename(columns={"energy_well_width": "width"})[
            ["epsilon", "run", "layer", "width"]
        ]

    curves_path = ENERGY_POTENTIAL_CURVES_V1 if ENERGY_POTENTIAL_CURVES_V1.exists() else ENERGY_POTENTIAL_CURVES
    if not curves_path.exists():
        raise FileNotFoundError(
            "Energy potential CSVs not found. Run figure13_layer_energy_potential_v1.py first."
        )
    curves = pd.read_csv(curves_path)
    rows = []
    for (eps, layer, run), sub in curves.groupby(["epsilon", "layer", "run"]):
        width = _well_width(
            sub["dynamic_fc_state"].to_numpy(dtype=float),
            sub["potential"].to_numpy(dtype=float),
            dU,
        )
        rows.append({"epsilon": float(eps), "layer": layer, "run": run, "width": width})
    width_df = pd.DataFrame(rows)

    stats_rows = []
    for (layer, eps), grp in width_df.groupby(["layer", "epsilon"]):
        vals = grp["width"].dropna()
        stats_rows.append({
            "figure": "figure13",
            "panel": "B",
            "metric": "energy_well_width",
            "test": f"U(z) <= U_min + {dU:g}",
            "layer": layer,
            "epsilon": float(eps),
            "mean_width": float(vals.mean()) if len(vals) else np.nan,
            "sem_width": float(vals.sem()) if len(vals) > 1 else np.nan,
            "n": int(len(vals)),
        })
    return width_df, stats_rows


def compute_fcv_slopes(eps, layer_temporal_std_fc):
    """Per-run linear slope dFCV/deps for each layer. Returns (n_runs, n_layers)."""
    arr = np.asarray(layer_temporal_std_fc, dtype=float)
    eps = np.asarray(eps, dtype=float)
    n_eps, n_runs, n_layers = arr.shape
    slopes = np.full((n_runs, n_layers), np.nan)
    for r in range(n_runs):
        for layer_idx in range(n_layers):
            y = arr[:, r, layer_idx]
            mask = np.isfinite(y) & np.isfinite(eps)
            if mask.sum() >= 2:
                slopes[r, layer_idx] = np.polyfit(eps[mask], y[mask], 1)[0]
    return slopes


def compute_width_slopes(width_df):
    """Per-run linear slope dWidth/deps for each layer. Returns (slopes, layer_names).

    slopes has shape (n_runs, n_layers) with layers ordered L1..Ln.
    """
    def _layer_num(name):
        try:
            return int(str(name).split()[-1])
        except (ValueError, IndexError):
            return 0

    layer_names = sorted(width_df["layer"].unique(), key=_layer_num)
    per_layer = []
    n_runs = 0
    for layer_name in layer_names:
        grp = width_df[width_df["layer"].eq(layer_name)]
        pivot = grp.pivot_table(index="run", columns="epsilon", values="width").sort_index(axis=1)
        eps = pivot.columns.to_numpy(dtype=float)
        run_slopes = []
        for _, row in pivot.iterrows():
            y = row.to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(eps)
            run_slopes.append(np.polyfit(eps[mask], y[mask], 1)[0] if mask.sum() >= 2 else np.nan)
        per_layer.append(np.asarray(run_slopes, dtype=float))
        n_runs = max(n_runs, len(run_slopes))

    slopes = np.full((n_runs, len(layer_names)), np.nan)
    for layer_idx, col in enumerate(per_layer):
        slopes[: len(col), layer_idx] = col
    return slopes, [f"L{_layer_num(name)}" for name in layer_names]


def compute_fcv_slopes_v1(summary_path=ENERGY_POTENTIAL_SUMMARY_V1,
                          value_col="fcv_from_fc_state"):
    """Per-run dFCV/deps from the 50-run v1 summary (same simulation as width).

    Reuses the width-slope pivot/regression by aliasing the FCV column, so panel
    B's FCV shares the width source and sample size (n=50, all layers).
    """
    df = pd.read_csv(summary_path)
    tmp = df.rename(columns={value_col: "width"})[["epsilon", "run", "layer", "width"]]
    return compute_width_slopes(tmp)


def compute_fcv_endpoint_delta(summary_path=ENERGY_POTENTIAL_SUMMARY_V1,
                               value_col="fcv_from_fc_state"):
    """Per-run ΔFCV/Δε as the endpoint difference (FCV(ε_max) − FCV(ε_min)) / Δε.

    Uses only the two endpoint ε (Δε = 1 here), which separates L1 from L2 far
    better than the full-range OLS slope. Returns (deltas (n_runs, n_layers), labels).
    """
    def _layer_num(name):
        try:
            return int(str(name).split()[-1])
        except (ValueError, IndexError):
            return 0

    df = pd.read_csv(summary_path)
    layer_names = sorted(df["layer"].unique(), key=_layer_num)
    eps_lo, eps_hi = float(df["epsilon"].min()), float(df["epsilon"].max())
    d_eps = eps_hi - eps_lo
    per_layer, n_runs = [], 0
    for layer_name in layer_names:
        grp = df[df["layer"].eq(layer_name)]
        pivot = grp.pivot_table(index="run", columns="epsilon", values=value_col)
        delta = (pivot[eps_hi] - pivot[eps_lo]).to_numpy(dtype=float) / d_eps
        per_layer.append(delta)
        n_runs = max(n_runs, len(delta))
    deltas = np.full((n_runs, len(layer_names)), np.nan)
    for layer_idx, col in enumerate(per_layer):
        deltas[: len(col), layer_idx] = col
    return deltas, [f"L{_layer_num(name)}" for name in layer_names]


def load_figure10_large_scale_fcv():
    """Prepare the v1 bootstrap FCV-difference distributions for Panel E."""
    observations = pd.read_csv(RAW_DIR / "large_scale_fcv_observations.csv")
    expected = {"Base", "NULL-Out", "NULL-In"}
    if set(observations["condition"]) != expected:
        raise RuntimeError("Large-scale FCV table must contain Base, NULL-Out, and NULL-In")
    base = observations.loc[observations["condition"].eq("Base"), "FCV"].to_numpy(float)
    null_out = observations.loc[
        observations["condition"].eq("NULL-Out"), "FCV"
    ].to_numpy(float)
    null_in = observations.loc[
        observations["condition"].eq("NULL-In"), "FCV"
    ].to_numpy(float)
    groups = [base, null_out, null_in]
    np.random.seed(0)
    fcv_boot, fcv_p, fcv_rows = run_three_bootstrap(
        null_out, null_in, base, n_boot=10000, holm_swap=False
    )
    stats_rows = []
    fcv_kruskal = kruskal(*groups)
    stats_rows.append({
        "figure": "figure13",
        "panel": "E",
        "metric": "FCV",
        "test": "Kruskal-Wallis",
        "comparison": "Base vs NULL-Out vs NULL-In",
        "statistic": fcv_kruskal.statistic,
        "p_value": fcv_kruskal.pvalue,
        "n_base": len(base),
        "n_null_out": len(null_out),
        "n_null_in": len(null_in),
        "mean_base": np.mean(base),
        "mean_null_out": np.mean(null_out),
        "mean_null_in": np.mean(null_in),
    })

    for row in fcv_rows:
        row.update({
            "figure": "figure13",
            "panel": "E",
            "metric": "FCV",
            "test": "bootstrap mean difference",
        })
        stats_rows.append(row)
    return list(fcv_boot), list(fcv_p), stats_rows


def prepare_plot_data():
    results, args = load_figure11_layer_data()
    fcv_boot, fcv_p, stats_rows = load_figure10_large_scale_fcv()

    # Panels B and C use the same dense-epsilon simulations and the same FCV
    # definition: temporal SD of each layer's mean FC with the other layers.
    dense_fcv = pd.read_csv(ENERGY_POTENTIAL_SUMMARY_V1)
    required_dense = {"epsilon", "run", "layer", "fcv_from_fc_state"}
    missing_dense = required_dense.difference(dense_fcv.columns)
    if missing_dense:
        raise KeyError(f"Dense layer-FCV table lacks columns: {sorted(missing_dense)}")

    def _layer_number(label):
        return int(str(label).split()[-1])

    dense_layers = sorted(dense_fcv["layer"].unique(), key=_layer_number)
    dense_stats = (
        dense_fcv.groupby(["epsilon", "layer"])["fcv_from_fc_state"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    std_mean_df = dense_stats.pivot(index="epsilon", columns="layer", values="mean")
    std_sem_df = dense_stats.pivot(index="epsilon", columns="layer", values="sem")
    std_mean_df = std_mean_df.reindex(columns=dense_layers).sort_index()
    std_sem_df = std_sem_df.reindex(index=std_mean_df.index, columns=dense_layers)
    eps = std_mean_df.index.to_numpy(dtype=float)
    std_mean = std_mean_df.to_numpy(dtype=float)
    std_sem = std_sem_df.to_numpy(dtype=float)

    fcv_slopes, fcv_layer_labels = compute_fcv_slopes_v1()
    stats_rows.extend(
        _slope_stats_rows(fcv_slopes, fcv_layer_labels, panel="C",
                          metric="fcv_slope", test="linear slope dFCV/deps, one-sample t vs 0")
    )

    # Panel C: paired L1-vs-other-layer comparisons (Wilcoxon + Holm).
    fcv_l1_pvals, fcv_pair_rows = _l1_pairwise_stats(
        fcv_slopes, fcv_layer_labels, panel="C", metric="fcv_slope")
    stats_rows.extend(fcv_pair_rows)

    return {
        "args": args,
        "eps": eps,
        "std_mean": std_mean,
        "std_sem": std_sem,
        "fcv_boot": fcv_boot,
        "fcv_p": fcv_p,
        "fcv_slopes": fcv_slopes,
        "fcv_layer_labels": fcv_layer_labels,
        "fcv_l1_pvals": fcv_l1_pvals,
        "stats_rows": stats_rows,
    }


def _slope_stats_rows(slopes, layer_labels, panel, metric, test):
    """One-sample-t summary rows (mean, sem, t, p) per layer for a slope matrix."""
    rows = []
    for layer_idx, label in enumerate(layer_labels):
        vals = slopes[:, layer_idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            t_stat, p_val = ttest_1samp(vals, 0.0)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            "figure": "figure13",
            "panel": panel,
            "metric": metric,
            "test": test,
            "layer": label,
            "mean_slope": float(np.mean(vals)) if len(vals) else np.nan,
            "sem_slope": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "n": int(len(vals)),
        })
    return rows


def _l1_pairwise_stats(slopes, layer_labels, panel, metric):
    """L1 vs each other layer: PAIRED Wilcoxon signed-rank, Holm-corrected across
    the (n_layers-1) comparisons.

    Layers within a run share the same simulation (same seed), so per-run slopes
    are matched across layers -- a paired test is the correct choice and is far
    more powerful than treating layers as independent samples.
    Returns (holm_pvals list aligned to L2..Ln, rows).
    """
    raw_p, others = [], []
    for j in range(1, slopes.shape[1]):
        a, b = slopes[:, 0], slopes[:, j]
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        w_stat, p = wilcoxon(a, b)
        raw_p.append(p)
        others.append((j, w_stat, a, b))
    holm_p = multipletests(raw_p, method="holm")[1] if raw_p else []
    rows = []
    for (j, w_stat, a, b), p_raw, p_holm in zip(others, raw_p, holm_p):
        rows.append({
            "figure": "figure13",
            "panel": panel,
            "metric": metric,
            "test": "Wilcoxon signed-rank (paired by run), Holm-corrected",
            "comparison": f"{layer_labels[0]} vs {layer_labels[j]}",
            "statistic": float(w_stat),
            "p_value": float(p_raw),
            "p_holm": float(p_holm),
            "mean_L1": float(np.mean(a)) if len(a) else np.nan,
            "mean_other": float(np.mean(b)) if len(b) else np.nan,
            "mean_paired_diff": float(np.mean(a - b)) if len(a) else np.nan,
            "n_pairs": int(len(a)),
        })
    return list(map(float, holm_p)), rows


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

def draw_energy_width_panel(ax, width_df, highlight_layer=1):
    """Panel B: energy well width vs epsilon for every layer, highlighting Layer 1.

    Layer 1 is drawn bold with a shaded SEM band; the deeper layers are muted so the
    reader sees at a glance that the top (input) layer has the widest, most
    epsilon-sensitive potential well.
    """
    layer_colors = [
        fig_help.ZEBRAFISH_DIVISION_COLORS["Tel"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Di"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Mes"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Hind"],
    ]

    def _layer_num(name):
        try:
            return int(str(name).split()[-1])
        except (ValueError, IndexError):
            return 0

    layer_names = sorted(width_df["layer"].unique(), key=_layer_num)
    hi_curve = None
    # Draw muted (background) layers first, highlighted layer last so it sits on top.
    for is_highlight in (False, True):
        for layer_name in layer_names:
            num = _layer_num(layer_name)
            if (num == highlight_layer) != is_highlight:
                continue
            grp = width_df[width_df["layer"].eq(layer_name)].dropna(subset=["width"])
            if grp.empty:
                continue
            color = layer_colors[(num - 1) % len(layer_colors)]
            agg = grp.groupby("epsilon")["width"].agg(["mean", "sem"]).sort_index()
            x = agg.index.to_numpy(dtype=float)
            mean = agg["mean"].to_numpy(dtype=float)
            sem = agg["sem"].to_numpy(dtype=float)
            if is_highlight:
                ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.20,
                                linewidth=0, zorder=4)
                ax.errorbar(x, mean, yerr=sem, marker="o", markersize=MARKER_SIZE * 1.1,
                            linewidth=LINE_W * 1.9, color=color, capsize=1.8,
                            elinewidth=0.9, zorder=6, label=f"L{num}")
                hi_curve = (x, mean, color, num)
            else:
                ax.plot(x, mean, marker="o", markersize=MARKER_SIZE * 0.6,
                        linewidth=LINE_W * 0.9, color=color, alpha=0.40,
                        zorder=2, label=f"L{num}")

    # Inline callout on the highlighted layer.
    if hi_curve is not None:
        x, mean, color, num = hi_curve
        ax.annotate(
            f"L{num}",
            xy=(x[-1], mean[-1]),
            xytext=(-2, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=color,
            fontweight="bold",
            fontsize=LEGEND_FS,
        )

    ax.set_xlabel(r"$\epsilon$", fontsize=AXIS_FS)
    ax.set_ylabel("Energy well width", fontsize=AXIS_FS)
    ax.set_title("")
    ax.grid(False)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(-0.08, 1.08)
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    leg = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        fontsize=LEGEND_FS,
        handlelength=1.2,
        labelspacing=0.25,
        borderaxespad=0.2,
        markerscale=0.7,
    )
    for text in leg.get_texts():
        if text.get_text() == f"L{highlight_layer}":
            text.set_fontweight("bold")


def draw_slope_bars(ax, slopes, ylabel, layer_labels=None, highlight_layer=None,
                    ylabel_fs=None):
    """Per-layer slope as bars with SEM and significance stars.

    Shared by panel B (dWidth/deps) and panel C (dFCV/deps). If highlight_layer is
    given, that bar gets a bold black edge and a bold x-tick label.
    """
    layer_colors = [
        fig_help.ZEBRAFISH_DIVISION_COLORS["Tel"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Di"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Mes"],
        fig_help.ZEBRAFISH_DIVISION_COLORS["Hind"],
    ]
    n_layers = slopes.shape[1]
    positions = np.arange(n_layers)
    if layer_labels is None:
        layer_labels = [f"L{i + 1}" for i in range(n_layers)]
    means = np.full(n_layers, np.nan)
    sems = np.full(n_layers, np.nan)
    stars = []
    for layer_idx in range(n_layers):
        vals = slopes[:, layer_idx]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            means[layer_idx] = np.mean(vals)
        if len(vals) > 1:
            sems[layer_idx] = np.std(vals, ddof=1) / np.sqrt(len(vals))
            _, p_val = ttest_1samp(vals, 0.0)
            stars.append(_sig_star(p_val))
        else:
            stars.append("")

    highlight_label = f"L{highlight_layer}" if highlight_layer is not None else None
    edge_widths = [1.5 if lbl == highlight_label else 0.6 for lbl in layer_labels]
    ax.bar(
        positions,
        means,
        yerr=sems,
        width=0.68,
        color=[layer_colors[i % len(layer_colors)] for i in range(n_layers)],
        edgecolor="black",
        linewidth=edge_widths,
        error_kw={"elinewidth": 0.8, "capsize": 2},
        zorder=3,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    finite_means = means[np.isfinite(means)]
    finite_sems = np.nan_to_num(sems)
    top = np.nanmax(means + finite_sems) if len(finite_means) else 1.0
    y_step = (top if top > 0 else 1.0) * 0.06
    for pos, mean, sem, star in zip(positions, means, sems, stars):
        if star and np.isfinite(mean):
            ax.text(pos, mean + (0 if np.isnan(sem) else sem) + y_step, star,
                    ha="center", va="bottom", fontsize=STAR_FS)

    ax.set_xticks(positions)
    ax.set_xticklabels(layer_labels, rotation=0, ha="center", fontsize=TICK_FS)
    if highlight_label is not None:
        for tick in ax.get_xticklabels():
            if tick.get_text() == highlight_label:
                tick.set_fontweight("bold")
    ax.set_ylabel(ylabel, fontsize=ylabel_fs if ylabel_fs is not None else AXIS_FS)
    ax.set_title("")
    ax.grid(False)
    ax.set_xlim(-0.7, n_layers - 0.3)
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _slope_mean_sem_stars(slopes):
    """Return per-layer (means, sems, stars) for a (n_runs, n_layers) array."""
    n_layers = slopes.shape[1]
    means = np.full(n_layers, np.nan)
    sems = np.full(n_layers, np.nan)
    stars = []
    for layer_idx in range(n_layers):
        vals = slopes[:, layer_idx]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            means[layer_idx] = np.mean(vals)
        if len(vals) > 1:
            sems[layer_idx] = np.std(vals, ddof=1) / np.sqrt(len(vals))
            _, p_val = ttest_1samp(vals, 0.0)
            stars.append(_sig_star(p_val))
        else:
            stars.append("")
    return means, sems, stars


def draw_merged_slope_bars(ax, width_slopes, fcv_slopes, layer_labels=None):
    """Panel B: per-layer dWidth/deps and dFCV/deps as grouped bars on a twin
    y-axis (left = Width, right = FCV). Both metrics share the L1-L4 x-groups;
    each axis is colour-matched to its series so the differing scales are clear.
    """
    width_color = fig_help.MAIN_COLORS["pre_in"]   # blue  -> left axis
    fcv_color = fig_help.MAIN_COLORS["post_out"]    # orange -> right axis

    n_layers = width_slopes.shape[1]
    if layer_labels is None:
        layer_labels = [f"L{i + 1}" for i in range(n_layers)]
    positions = np.arange(n_layers)
    bar_w = 0.38

    w_means, w_sems, w_stars = _slope_mean_sem_stars(width_slopes)
    f_means, f_sems, f_stars = _slope_mean_sem_stars(fcv_slopes)

    ax2 = ax.twinx()

    ax.bar(
        positions - bar_w / 2, w_means, yerr=w_sems, width=bar_w,
        color=width_color, edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 0.8, "capsize": 2}, zorder=3, label=r"$\Delta$Width",
    )
    ax2.bar(
        positions + bar_w / 2, f_means, yerr=f_sems, width=bar_w,
        color=fcv_color, edgecolor="black", linewidth=0.6,
        error_kw={"elinewidth": 0.8, "capsize": 2}, zorder=3, label=r"$\Delta$FCV",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    # Both metrics are positive; anchor both axes at 0 so bars share a baseline.
    w_top = np.nanmax(w_means + np.nan_to_num(w_sems))
    f_top = np.nanmax(f_means + np.nan_to_num(f_sems))
    #ax.set_ylim(0, w_top * 1.28)
    #ax2.set_ylim(0, f_top * 1.28)

    for pos, mean, sem, star in zip(positions - bar_w / 2, w_means, w_sems, w_stars):
        if star and np.isfinite(mean):
            ax.text(pos, mean + (0 if np.isnan(sem) else sem) + w_top * 0.04, star,
                    ha="center", va="bottom", fontsize=STAR_FS, color=width_color)
    for pos, mean, sem, star in zip(positions + bar_w / 2, f_means, f_sems, f_stars):
        if star and np.isfinite(mean):
            ax2.text(pos, mean + (0 if np.isnan(sem) else sem) + f_top * 0.04, star,
                     ha="center", va="bottom", fontsize=STAR_FS, color=fcv_color)

    ax.set_xticks(positions)
    ax.set_xticklabels(layer_labels, rotation=0, ha="center", fontsize=TICK_FS)
    ax.set_xlim(-0.7, n_layers - 0.3)

    ax.set_ylabel(r"$\Delta$Width$\,/\,\Delta\epsilon$", fontsize=AXIS_FS, color=width_color)
    ax2.set_ylabel(r"$\Delta$FCV$\,/\,\Delta\epsilon$", fontsize=AXIS_FS, color=fcv_color)
    ax.tick_params(axis="y", labelsize=TICK_FS, colors=width_color)
    ax2.tick_params(axis="y", labelsize=TICK_FS, colors=fcv_color)
    ax.tick_params(axis="x", labelsize=TICK_FS, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.grid(False)
    ax2.grid(False)
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.spines["left"].set_color(width_color)
    ax.spines["left"].set_linewidth(1.0)
    ax2.spines["right"].set_color(fcv_color)
    ax2.spines["right"].set_linewidth(1.0)
    ax2.spines["right"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.8)

    handles = [
        mpatches.Patch(facecolor=width_color, edgecolor="black", linewidth=0.6, label=r"$\Delta$Width$/\Delta\epsilon$"),
        mpatches.Patch(facecolor=fcv_color, edgecolor="black", linewidth=0.6, label=r"$\Delta$FCV$/\Delta\epsilon$"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False,
              fontsize=TICK_FS, handlelength=1.1, handletextpad=0.4, labelspacing=0.25)


def _violin_box(ax, pos, data, box_w=0.06):
    """Overlay a whisker line + IQR box + median dot (panel-D violin style)."""
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    wlo = max(np.min(data), q1 - 1.5 * iqr)
    whi = min(np.max(data), q3 + 1.5 * iqr)
    ax.plot([pos, pos], [wlo, whi], color="black", linewidth=0.8, zorder=3)
    rect = plt.Rectangle((pos - box_w / 2, q1), box_w, iqr, facecolor="white",
                         edgecolor="black", linewidth=0.7, zorder=4)
    ax.add_patch(rect)
    ax.scatter([pos], [med], color="white", s=12, zorder=5,
               edgecolors="black", linewidths=0.7)


def _draw_slope_violins(ax, slopes, positions, color, box_w=0.06):
    """Per-layer per-run slope distributions as violins (+ box) on `ax`.

    Returns (per-layer max value, per-layer significance stars).
    """
    cleaned = [slopes[:, i][np.isfinite(slopes[:, i])] for i in range(slopes.shape[1])]
    parts = ax.violinplot(cleaned, positions=positions, widths=0.34,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")
    tops, stars = [], []
    for pos, vals in zip(positions, cleaned):
        _violin_box(ax, pos, vals, box_w=box_w)
        tops.append(np.max(vals))
        _, p_val = ttest_1samp(vals, 0.0) if len(vals) > 1 else (np.nan, np.nan)
        stars.append(_sig_star(p_val))
    return np.asarray(tops), stars


def _add_pairwise_brackets(ax, x_ref, x_others, pvals, color, y0, step=0.065, h=0.016):
    """L1-vs-other significance brackets in blended (x=data, y=axes-fraction)
    coords so they sit in the panel headroom regardless of the twin-axis scale.
    """
    trans = ax.get_xaxis_transform()
    for k, (xo, p) in enumerate(zip(x_others, pvals)):
        y = y0 + k * step
        ax.plot([x_ref, x_ref, xo, xo], [y, y + h, y + h, y], transform=trans,
                color=color, linewidth=0.9, clip_on=False, zorder=7)
        ax.text((x_ref + xo) / 2, y + h, _sig_star(p), transform=trans,
                ha="center", va="bottom", fontsize=STAR_FS, color=color, clip_on=False)


def draw_fcv_slope_violins(ax, fcv_slopes, layer_labels=None, fcv_l1_pvals=None):
    """Panel B: per-layer dFCV/deps violins with L1-vs-other comparisons."""
    fcv_color = fig_help.MAIN_COLORS["post_out"]

    n_layers = fcv_slopes.shape[1]
    if layer_labels is None:
        layer_labels = [f"L{i + 1}" for i in range(n_layers)]
    base = np.arange(n_layers)
    _draw_slope_violins(ax, fcv_slopes, base, fcv_color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

    # Anchor the scale at zero and reserve headroom for pairwise brackets.
    f_vals = fcv_slopes[np.isfinite(fcv_slopes)]
    f_lo = float(np.min(f_vals))
    f_hi = float(np.max(f_vals))
    f_range = max(f_hi - f_lo, 1e-9)
    ax.set_ylim(min(0.0, f_lo) - 0.02 * f_range, f_hi + 0.16 * f_range)

    # L1-vs-other Holm-corrected brackets.
    if fcv_l1_pvals is not None:
        _add_pairwise_brackets(ax, base[0], base[1:], fcv_l1_pvals,
                               fcv_color, y0=0.80)

    ax.set_xticks(base)
    ax.set_xticklabels(layer_labels, rotation=0, ha="center", fontsize=TICK_FS)
    ax.set_xlim(-0.7, n_layers - 0.3)

    ax.set_ylabel(r"$\Delta$FCV$\,/\,\Delta\epsilon$", fontsize=AXIS_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.grid(False)
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)


def draw_condition_fcv_violins(ax, groups, pair_indices, p_holm):
    """Panel E: original FCV distributions with all Holm-corrected contrasts."""
    labels = ["Base", "NULL-Out", "NULL-In"]
    colors = [
        fig_help.MAIN_COLORS["neutral_light"],
        fig_help.MAIN_COLORS["post_out"],
        fig_help.MAIN_COLORS["pre_in"],
    ]
    positions = np.arange(len(groups), dtype=float)
    clean = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in groups]
    parts = ax.violinplot(
        clean,
        positions=positions,
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.65)
    for pos, values in zip(positions, clean):
        _violin_box(ax, pos, values, box_w=0.12)

    all_values = np.concatenate(clean)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    y_range = max(y_max - y_min, 1e-9)
    # Keep all observations visible while tightening the vertical range so the
    # three original FCV distributions occupy more of Panel E.
    ax.set_ylim(y_min - 0.01 * y_range, y_max + 0.04 * y_range)

    # Display every pairwise Mann-Whitney comparison after Holm correction.
    trans = ax.get_xaxis_transform()
    for level, ((i, j), p_value) in enumerate(zip(pair_indices, p_holm)):
        y = 0.77 + level * 0.075
        h = 0.018
        ax.plot(
            [positions[i], positions[i], positions[j], positions[j]],
            [y, y + h, y + h, y],
            transform=trans,
            color="#333333",
            linewidth=0.9,
            clip_on=False,
            zorder=7,
        )
        ax.text(
            (positions[i] + positions[j]) / 2,
            y + h,
            _sig_star(p_value),
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=STAR_FS,
            color="#222222",
            clip_on=False,
        )

    ax.set_xlim(-0.65, 2.65)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_ylabel("FCV", fontsize=AXIS_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


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
    # Top: layer schematic, FCV-epsilon curves, and FCV slopes. Bottom:
    # whole-brain rewiring schematic and its FCV perturbation result.
    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(
        2,
        6,
        figure=fig,
        left=0.065,
        right=0.965,
        top=0.90,
        bottom=0.075,
        wspace=0.75,
        hspace=0.40,
        height_ratios=[1.0, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    # Match the lower-row panel widths to panels A and B (two grid columns
    # each) and center the D/E pair within the six-column figure grid.
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2:4])

    # Reduce the widths of the quantitative panels while keeping each one
    # centered within its original GridSpec allocation.
    for ax in (ax_b, ax_c, ax_e):
        pos = ax.get_position()
        reduced_width = pos.width * 0.85
        ax.set_position([
            pos.x0 + (pos.width - reduced_width) / 2,
            pos.y0,
            reduced_width,
            pos.height,
        ])

    axes = {
        "a": ax_a,
        "b": ax_b,
        "c": ax_c,
        "d": ax_d,
        "e": ax_e,
    }
    return fig, axes


# 3. Draw each panel

def draw_panel_a(ax_a, plot_data):
    draw_layer_network_panel_clean(ax_a, plot_data["args"])


def draw_panel_b(ax_b, plot_data):
    # Same FCV-versus-epsilon panel shown as panel B in figure_supply_15.png.
    draw_layer_fc_panel(
        ax_b,
        plot_data["eps"],
        plot_data["std_mean"],
        plot_data["std_sem"],
        plot_data["args"],
        "FCV",
        "Layer-mean std FC",
        show_legend=True,
    )


def draw_panel_c(ax_c, plot_data):
    # Per-layer dFCV/deps with L1-vs-other significance brackets.
    draw_fcv_slope_violins(
        ax_c,
        plot_data["fcv_slopes"],
        layer_labels=plot_data["fcv_layer_labels"],
        fcv_l1_pvals=plot_data["fcv_l1_pvals"],
    )


def draw_panel_d(ax_d):
    draw_large_scale_model_panel(ax_d)


def draw_panel_e(ax_e, plot_data):
    large_scale_colors = [
        fig_help.MAIN_COLORS["post_out"],
        fig_help.MAIN_COLORS["pre_in"],
        fig_help.MAIN_COLORS["neutral_light"],
    ]
    plot_violin_bootstrap(
        ax_e,
        plot_data["fcv_boot"],
        plot_data["fcv_p"],
        r"$\Delta$FCV",
        large_scale_colors,
    )
    ax_e.yaxis.label.set_color(fig_help.MAIN_COLORS["neutral_dark"])


def draw_all_panels(axes, plot_data):
    draw_panel_a(axes["a"], plot_data)
    draw_panel_b(axes["b"], plot_data)
    draw_panel_c(axes["c"], plot_data)
    draw_panel_d(axes["d"])
    draw_panel_e(axes["e"], plot_data)


# 4. Panel position adjustment and panel labels

def adjust_panel_positions_and_labels(fig, axes):
    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]
    ax_e = axes["e"]

    fig.canvas.draw()

    # Row headers: top row (A-C) = layer model; bottom row (D,E) = whole-brain.
    top_left = ax_a.get_position().x0
    top_right = ax_c.get_position().x1
    bot_left = ax_d.get_position().x0
    bot_right = ax_e.get_position().x1
    top_header_y = max(ax_a.get_position().y1, ax_b.get_position().y1, ax_c.get_position().y1) + 0.045
    bot_header_y = max(ax_d.get_position().y1, ax_e.get_position().y1) + 0.045
    fig.text(
        (top_left + top_right) / 2, top_header_y, "Layer linear model",
        ha="center", va="bottom", fontsize=TITLE_FS, fontweight="bold",
    )
    fig.text(
        (bot_left + bot_right) / 2, bot_header_y, "Zebrafish whole-brain network model",
        ha="center", va="bottom", fontsize=TITLE_FS, fontweight="bold",
    )

    label_positions = {
        "A": (-0.10, 1.05),
        "B": (-0.35, 1.05),
        "C": (-0.41, 1.05),
        "D": (-0.10, 1.05),
        "E": (-0.35, 1.05),
    }
    for ax, label in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (ax_e, "E")
    ]:
        x, y = label_positions[label]
        ax.text(
            x, y, label, transform=ax.transAxes,
            fontsize=PANEL_FS, fontweight="bold", ha="left", va="bottom",
        )


# 5. Save figure and statistics

def save_figure(fig):
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=False)


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
