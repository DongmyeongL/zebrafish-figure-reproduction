"""Empirical energy potential for the layer asymmetric linear model.

The plot uses the same dynamic-FC state that underlies FCV. For each layer,
z_l(w) is the sliding-window mean FC from that layer to all other layers. We
estimate P(z_l) and draw U_l(z) = -log P(z_l), shifted so each minimum is zero.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from numba import njit


PACK_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
DATA = DERIVED_ROOT / "figure13" / "layer_asymmetric_epsilon_linear_data.npz"
FIG = PACK_ROOT / "figures"
TAB = DERIVED_ROOT / "figure13"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG / "figure13_layer_energy_potential.png"
OUT_CSV = TAB / "figure13_layer_energy_potential_curves.csv"
OUT_SUMMARY = TAB / "figure13_layer_energy_potential_summary.csv"

EPSILONS = [0.0, 0.5, 1.0]
N_RUNS = 50
N_STEPS = 60_000
BURN_FRAC = 0.15
N_BINS = 100
N_GRID = 180
SEED = 20260530
WINDOW_SEC = 3.0
STEP_SEC = 1.5
LAYER_COLORS = {"Layer 1": "#2B2B2B", "Layer 4": "#0072B2"}
EPSILON_COLORS = {0.0: "#2B2B2B", 0.5: "#D55E00", 1.0: "#0072B2"}
FORCE_RECOMPUTE = False

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import layer_linear_model as layer_model


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
    }
)


@njit(cache=True)
def simulate_layer_means_fast(
    j_mat: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    dt: float,
    noise_sigma: float,
    epsilon_slow_layer_drive: float,
    slow_drive_frequency: float,
    slow_layer_scales: np.ndarray,
    input_amplitude: float,
    input_frequency: float,
    slow_phases: np.ndarray,
    noise: np.ndarray,
    burn: int,
) -> np.ndarray:
    n_steps, n_nodes = noise.shape
    n_layers = len(starts)
    out = np.empty((n_steps - burn, n_layers), dtype=np.float64)
    state = np.zeros(n_nodes, dtype=np.float64)
    noise_scale = noise_sigma * np.sqrt(dt)

    for step in range(n_steps):
        t = step * dt
        dx = j_mat @ state

        if input_amplitude != 0.0:
            stimulus = np.sin(2.0 * np.pi * input_frequency * t)
            for node in range(starts[0], stops[0]):
                dx[node] += input_amplitude * stimulus

        if epsilon_slow_layer_drive != 0.0:
            for layer_idx in range(n_layers):
                slow_drive = (
                    epsilon_slow_layer_drive
                    * slow_layer_scales[layer_idx]
                    * np.sin(2.0 * np.pi * slow_drive_frequency * t + slow_phases[layer_idx])
                )
                for node in range(starts[layer_idx], stops[layer_idx]):
                    dx[node] += slow_drive

        for node in range(n_nodes):
            state[node] += dt * dx[node] + noise_scale * noise[step, node]

        if step >= burn:
            out_idx = step - burn
            for layer_idx in range(n_layers):
                total = 0.0
                count = stops[layer_idx] - starts[layer_idx]
                for node in range(starts[layer_idx], stops[layer_idx]):
                    total += state[node]
                out[out_idx, layer_idx] = total / count
    return out


def build_original_jacobian(params: dict[str, np.ndarray | float], epsilon: float) -> tuple[np.ndarray, list[slice]]:
    """Use the original Figure13 code for the network equation/J matrix."""
    return layer_model.build_layer_jacobian(
        layer_sizes=params["layer_sizes"].astype(int).tolist(),
        gamma=float(params["gamma"]),
        w_intra=float(params["w_intra"]),
        w_inter=float(params["w_inter"]),
        intra_epsilon=float(params["intra_epsilon"]),
        epsilon=epsilon,
        inter_epsilon_scales=params["inter_epsilon_scales"].astype(float).tolist(),
        layer_decay_offsets=params["layer_decay_offsets"].astype(float).tolist(),
    )


def simulate_layer_means_from_original_model(
    params: dict[str, np.ndarray | float], epsilon: float, seed: int
) -> np.ndarray:
    """Fast simulation of the original linear model, storing only layer means."""
    rng = np.random.default_rng(seed)
    j_mat, slices = build_original_jacobian(params, epsilon)
    starts = np.asarray([s.start for s in slices], dtype=np.int64)
    stops = np.asarray([s.stop for s in slices], dtype=np.int64)
    slow_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(slices)).astype(np.float64)
    noise = rng.standard_normal((N_STEPS, j_mat.shape[0])).astype(np.float64)
    return simulate_layer_means_fast(
        np.asarray(j_mat, dtype=np.float64),
        starts,
        stops,
        float(params["dt"]),
        float(params["noise_sigma"]),
        float(params["epsilon_slow_layer_drive"]),
        float(params["slow_drive_frequency"]),
        params["slow_layer_scales"].astype(np.float64),
        float(params["input_amplitude"]),
        float(params["input_frequency"]),
        slow_phases,
        noise,
        int(N_STEPS * BURN_FRAC),
    )


def smooth_density(density: np.ndarray) -> np.ndarray:
    x = np.arange(-3, 4, dtype=float)
    kernel = np.exp(-0.5 * (x / 1.2) ** 2)
    kernel /= kernel.sum()
    return np.convolve(density, kernel, mode="same")


def potential_curve(values: np.ndarray, bin_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bin_edges, density=True)
    density = smooth_density(density)
    density = np.maximum(density, 1e-9)
    centers = 0.5 * (edges[:-1] + edges[1:])
    potential = -np.log(density)
    potential -= np.nanmin(potential)
    return centers, potential


def robust_edges(values: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [0.1, 99.9])
    pad = 0.25 * max(hi - lo, 1e-6)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(values), axis=0))
    return mean, sem


def layer_dynamic_fc(layer_signals: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    window = max(8, int(round(WINDOW_SEC / dt)))
    step = max(1, int(round(STEP_SEC / dt)))
    fc_values = []
    for start in range(0, layer_signals.shape[0] - window + 1, step):
        segment = layer_signals[start : start + window]
        corr = np.corrcoef(segment.T)
        corr[~np.isfinite(corr)] = np.nan
        fc_values.append(corr)
    fc_stack = np.asarray(fc_values, dtype=float)
    layer_fc_state = np.column_stack(
        [
            np.nanmean(np.delete(fc_stack[:, layer_idx, :], layer_idx, axis=1), axis=1)
            for layer_idx in range(fc_stack.shape[1])
        ]
    )
    layer_fcv = np.nanstd(layer_fc_state, axis=0)
    return layer_fc_state, layer_fcv


def main() -> None:
    if FORCE_RECOMPUTE or not cached_tables_current():
        compute_and_save_tables()

    curves = pd.read_csv(OUT_CSV)
    summary = pd.read_csv(OUT_SUMMARY)
    plotted = prepare_plot_curves(curves)

    fig, axes = plt.subplots(2, 1, figsize=(3.7, 4.8), sharex=False, sharey=True)
    visible_max = max(
        np.nanpercentile(np.minimum(potential, 12.0), 98)
        for layer_curves in plotted.values()
        for _, _, potential, _ in layer_curves
    )
    y_upper = max(6.0, min(12.0, float(visible_max) * 1.08))
    for ax, label in zip(axes, ["Layer 1", "Layer 4"]):
        layer_x_min = min(float(np.nanmin(centers)) for _, centers, _, _ in plotted[label])
        layer_x_max = max(float(np.nanmax(centers)) for _, centers, _, _ in plotted[label])
        x_pad = 0.05 * max(layer_x_max - layer_x_min, 1e-6)
        for epsilon, centers, potential, sem in plotted[label]:
            y = np.minimum(potential, y_upper)
            y_low = np.maximum(0.0, np.minimum(potential - sem, y_upper))
            y_high = np.minimum(potential + sem, y_upper)
            ax.fill_between(centers, y_low, y_high, color=EPSILON_COLORS[float(epsilon)], alpha=0.15, lw=0)
            ax.plot(
                centers,
                y,
                color=EPSILON_COLORS[float(epsilon)],
                lw=1.7,
                label=rf"$\epsilon={epsilon:.1f}$",
            )
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xlabel("dynamic FC state")
        ax.set_xlim(layer_x_min - x_pad, layer_x_max + x_pad)
        ax.set_ylim(0.0, y_upper)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", color="#E5E5E5", lw=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        add_fcv_inset(ax, summary, label)
    for ax in axes:
        ax.set_ylabel(r"potential  $U(z)$")
    handles = [Line2D([0], [0], color=EPSILON_COLORS[eps], lw=2.0, label=rf"$\epsilon={eps:.1f}$") for eps in EPSILONS]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper right", title="asymmetry")
    fig.suptitle("Layer linear model: epsilon-dependent dynamic-FC potential", fontsize=9.0, fontweight="bold", y=0.99)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Read {OUT_CSV}")
    print(f"Read {OUT_SUMMARY}")


def compute_and_save_tables() -> None:
    npz = np.load(DATA)
    params = {key: npz[key] for key in npz.files}

    rows = []
    summary_rows = []
    fc_state_values: dict[str, dict[float, list[np.ndarray]]] = {
        "Layer 1": {epsilon: [] for epsilon in EPSILONS},
        "Layer 4": {epsilon: [] for epsilon in EPSILONS},
    }
    for epsilon in EPSILONS:
        print(f"epsilon {epsilon:.1f}: {N_RUNS} fast simulations")
        for run_idx in range(N_RUNS):
            seed = SEED + int(round(epsilon * 1000)) * 1000 + run_idx
            layer_means = simulate_layer_means_from_original_model(params, epsilon, seed=seed)
            layer_fc_state, layer_fcv = layer_dynamic_fc(layer_means, float(params["dt"]))
            fc_state_values["Layer 1"][epsilon].append(layer_fc_state[:, 0])
            fc_state_values["Layer 4"][epsilon].append(layer_fc_state[:, -1])

            for label, values, layer_idx in [
                ("Layer 1", layer_fc_state[:, 0], 0),
                ("Layer 4", layer_fc_state[:, -1], -1),
            ]:
                within_1 = values[
                    (values >= np.nanpercentile(values, 0.5))
                    & (values <= np.nanpercentile(values, 99.5))
                ]
                width = float(np.nanpercentile(within_1, 95) - np.nanpercentile(within_1, 5))
                fcv = float(layer_fcv[layer_idx])
                summary_rows.append(
                    {
                        "epsilon": epsilon,
                        "run": run_idx,
                        "layer": label,
                        "fc_state_mean": float(np.nanmean(values)),
                        "fc_state_std": float(np.nanstd(values, ddof=1)),
                        "fc_state_width_p95_p05": width,
                        "fcv_from_fc_state": fcv,
                    }
                )

    for label in ["Layer 1", "Layer 4"]:
        for epsilon in EPSILONS:
            all_values = np.concatenate(fc_state_values[label][epsilon])
            edges = robust_edges(all_values)
            x_grid = np.linspace(edges[0], edges[-1], N_GRID)
            for run_idx, values in enumerate(fc_state_values[label][epsilon]):
                centers, potential = potential_curve(values, edges)
                interp = np.interp(x_grid, centers, potential, left=np.nan, right=np.nan)
                interp -= np.nanmin(interp)
                rows.extend(
                    {
                        "epsilon": epsilon,
                        "run": run_idx,
                        "layer": label,
                        "dynamic_fc_state": float(x),
                        "potential": float(u),
                        "fc_state_mean": float(np.nanmean(values)),
                        "fc_state_std": float(np.nanstd(values, ddof=1)),
                    }
                    for x, u in zip(x_grid, interp)
                )

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)


def cached_tables_current() -> bool:
    if not OUT_CSV.exists() or not OUT_SUMMARY.exists():
        return False
    try:
        summary = pd.read_csv(OUT_SUMMARY, usecols=["epsilon", "run", "layer"])
    except Exception:
        return False
    expected = len(EPSILONS) * N_RUNS * 2
    if len(summary) != expected:
        return False
    counts = summary.groupby(["epsilon", "layer"])["run"].nunique()
    return bool((counts == N_RUNS).all() and len(counts) == len(EPSILONS) * 2)


def prepare_plot_curves(curves: pd.DataFrame) -> dict[str, list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]]:
    plotted: dict[str, list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]] = {"Layer 1": [], "Layer 4": []}
    for (label, epsilon), sub in curves.groupby(["layer", "epsilon"], sort=False):
        pivot = sub.pivot_table(index="run", columns="dynamic_fc_state", values="potential", aggfunc="mean")
        centers = pivot.columns.to_numpy(float)
        run_potentials = pivot.to_numpy(float)
        potential_mean, potential_sem = mean_sem(run_potentials)
        plotted[str(label)].append((float(epsilon), centers, potential_mean, potential_sem))
    for label in plotted:
        plotted[label] = sorted(plotted[label], key=lambda item: item[0])
    return plotted


def add_fcv_inset(ax: plt.Axes, summary: pd.DataFrame, label: str) -> None:
    sub = summary[summary["layer"].eq(label)].copy()
    stats = sub.groupby("epsilon")["fcv_from_fc_state"].agg(["mean", "sem"]).reindex(EPSILONS)
    text_lines = ["FCV"]
    for eps, row in stats.iterrows():
        text_lines.append(rf"$\epsilon={eps:.1f}$: {row['mean']:.2f}$\pm${row['sem']:.2f}")
    ax.text(
        0.03,
        0.05,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.5, alpha=0.82),
    )


if __name__ == "__main__":
    main()
