import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import figure_style as fig_help


fig_help.apply_supplement_figure_style()

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "raw_data" / "figure_supply_15"
LINEAR_DIR = BASE_DIR
MODEL_CODE_DIR = ROOT_DIR / "data_processing_code" / "figure_supply_15"
if str(LINEAR_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_DIR))

EMPIRICAL_MOS5_TRACE = DATA_DIR / "figure_supply_15_rsp_rmos5_trace.npz"
LAYER_DATA = DATA_DIR / "figure13_inputs" / "layer_asymmetric_epsilon_linear_data.npz"
LAYER_TRACE_CACHE = DATA_DIR / "figure_supply_15_layer_trace_cache.npz"

OUT_PNG = ROOT_DIR / "figures" / "figure_supply_15.png"
OUT_WIDTH_STATS = ROOT_DIR / "statistics" / "figure_supply_15_energy_width_stats.csv"
DENSE_ENERGY_WIDTH = (
    ROOT_DIR / "derived_data" / "figure13" / "layer_energy_width_dense.csv"
)

EPSILON_CASES = [0.0, 0.3, 1.0]
EPSILON_PLOT_ORDER = [1.0, 0.3, 0.0]
LAYER_TRACE_REALIZATION = 1
LAYER_N_STEPS_SCALE = 10
LAYER_TRACE_START_STEP = 10000
LAYER_TRACE_WINDOW_STEPS = 2500
EMPIRICAL_WINDOW_STEPS = 900
EMPIRICAL_CASES = [
    ("base", "Base"),
    ("null_in", "Null-In"),
    ("null_out", "Null-Out"),
]

def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


layer_plot = _load_module(
    LINEAR_DIR / "plot_layer_asymmetric_epsilon_linear_model.py",
    "figure_supply_15_layer_plot",
)
layer_analysis = _load_module(
    MODEL_CODE_DIR / "analyze_layer_asymmetric_epsilon_linear_model.py",
    "figure_supply_15_layer_analysis",
)
figure13 = _load_module(
    LINEAR_DIR / "figure_supply_15_figure13_panels.py",
    "figure_supply_15_figure13_clean",
)


def zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def smooth(values, window=31):
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < 3:
        return values
    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def subtract_fitted_slow_sine(values, time, frequency=0.1):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if values.size < 3 or time.size != values.size:
        return values - np.nanmean(values)

    centered_time = time - time[0]
    omega_time = 2.0 * np.pi * frequency * centered_time
    design = np.column_stack(
        [
            np.sin(omega_time),
            np.cos(omega_time),
            np.ones_like(centered_time),
        ]
    )
    valid = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    if np.count_nonzero(valid) < design.shape[1]:
        return values - np.nanmean(values)

    coeffs, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    fitted_slow = design @ coeffs
    return values - fitted_slow


def style_trace_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_empirical_trace():
    if not EMPIRICAL_MOS5_TRACE.exists():
        raise FileNotFoundError(
            f"Missing bundled compact trace file: {EMPIRICAL_MOS5_TRACE}. "
            "The public package uses this precomputed trace instead of the large raw simulation files."
        )
    return np.load(EMPIRICAL_MOS5_TRACE, allow_pickle=True)


def select_empirical_window(data, window_steps=EMPIRICAL_WINDOW_STEPS):
    n_time = len(data["t_axis"])
    start_min = min(300, max(0, n_time - window_steps - 1))
    stop_max = max(start_min + window_steps, n_time - 10)
    best_score = -np.inf
    best_start = start_min

    for start in range(start_min, max(start_min + 1, stop_max - window_steps), 25):
        stop = start + window_steps
        score = 0.0
        for key, _ in EMPIRICAL_CASES:
            for suffix in ("p", "selected"):
                trace = smooth(np.asarray(data[f"{key}_{suffix}"][start:stop], dtype=float))
                score += float(np.nanstd(trace))
        if score > best_score:
            best_score = score
            best_start = start

    best_stop = min(best_start + window_steps, n_time)
    return best_start, best_stop


def plot_empirical_trace(ax, data, key, title, window, show_legend=False):
    t_axis = np.asarray(data["t_axis"], dtype=float)
    y_p = zscore(smooth(data[f"{key}_p"]))
    y_selected = zscore(smooth(data[f"{key}_selected"]))

    start, end = window
    end = min(end, len(t_axis), len(y_p), len(y_selected))
    t_axis = t_axis[start:end]
    t_axis = t_axis - t_axis[0]
    y_p = y_p[start:end]
    y_selected = y_selected[start:end]

    ax.plot(t_axis, y_p, color="#c0392b", lw=0.45, label="SP")
    ax.plot(t_axis, y_selected, color="#2f5597", lw=0.45, label=str(data["selected_region_name"]))
    ax.set_title(title, pad=3)
    ax.set_xlim(t_axis[0], t_axis[-1])

    ymin = min(np.nanmin(y_p), np.nanmin(y_selected))
    ymax = max(np.nanmax(y_p), np.nanmax(y_selected))
    pad = 0.08 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=fig_help.TICK_FS_2COL - 2,
            handlelength=1.5,
            borderaxespad=0.1,
        )
    style_trace_axis(ax)


def simulate_layer_traces():
    _, args = layer_plot.load_results(LAYER_DATA)
    n_steps = int(args.n_steps * LAYER_N_STEPS_SCALE)
    seed_sequence = np.random.SeedSequence([int(args.seed), int(LAYER_TRACE_REALIZATION)])
    seeds = seed_sequence.generate_state(len(EPSILON_CASES), dtype=np.uint32)

    cache = {
        "epsilon": np.asarray(EPSILON_CASES, dtype=float),
        "time": None,
        "l1": [],
        "l4": [],
    }
    for epsilon, seed in zip(EPSILON_CASES, seeds):
        time, _, signals, _, slices = layer_analysis.simulate_layer_network(
            epsilon=epsilon,
            layer_sizes=args.layer_sizes,
            n_steps=n_steps,
            dt=args.dt,
            gamma=args.gamma,
            w_intra=args.w_intra,
            w_inter=args.w_inter,
            intra_epsilon=args.intra_epsilon,
            inter_epsilon_scales=args.inter_epsilon_scales,
            layer_decay_offsets=args.layer_decay_offsets,
            noise_sigma=args.noise_sigma,
            epsilon_slow_layer_drive=args.epsilon_slow_layer_drive,
            slow_drive_frequency=args.slow_drive_frequency,
            slow_layer_scales=args.slow_layer_scales,
            input_amplitude=args.input_amplitude,
            input_frequency=args.input_frequency,
            seed=int(seed),
        )
        if cache["time"] is None:
            cache["time"] = time
        cache["l1"].append(np.mean(signals[:, slices[0]], axis=1))
        cache["l4"].append(np.mean(signals[:, slices[-1]], axis=1))

    np.savez(
        LAYER_TRACE_CACHE,
        epsilon=cache["epsilon"],
        time=np.asarray(cache["time"], dtype=float),
        l1=np.asarray(cache["l1"], dtype=float),
        l4=np.asarray(cache["l4"], dtype=float),
        n_steps=np.asarray(n_steps, dtype=int),
        realization=np.asarray(LAYER_TRACE_REALIZATION, dtype=int),
    )


def load_layer_traces():
    if not LAYER_TRACE_CACHE.exists():
        simulate_layer_traces()
    data = np.load(LAYER_TRACE_CACHE)
    epsilon = np.asarray(data["epsilon"], dtype=float)
    expected_steps = int(layer_plot.load_results(LAYER_DATA)[1].n_steps * LAYER_N_STEPS_SCALE)
    cached_steps = int(data["n_steps"]) if "n_steps" in data else int(data["time"].shape[0])
    cached_realization = int(data["realization"]) if "realization" in data else -1
    if (
        epsilon.shape[0] != len(EPSILON_CASES)
        or not np.allclose(epsilon, EPSILON_CASES)
        or cached_steps != expected_steps
        or cached_realization != LAYER_TRACE_REALIZATION
    ):
        simulate_layer_traces()
        data = np.load(LAYER_TRACE_CACHE)
    return data


def plot_layer_trace(ax, trace_data, idx, title, show_legend=False):
    time = np.asarray(trace_data["time"], dtype=float)
    l1 = zscore(subtract_fitted_slow_sine(trace_data["l1"][idx], time))
    l4 = zscore(subtract_fitted_slow_sine(trace_data["l4"][idx], time))

    start = min(LAYER_TRACE_START_STEP, max(0, len(time) - LAYER_TRACE_WINDOW_STEPS))
    end = min(start + LAYER_TRACE_WINDOW_STEPS, len(time), len(l1), len(l4))
    time = time[start:end] - time[start]
    l1 = l1[start:end]
    l4 = l4[start:end]

    ax.plot(time, l1, color="#c0392b", lw=0.45, label="L1")
    ax.plot(time, l4, color="#2f5597", lw=0.45, label="L4")
    ax.set_title(title, pad=3)
    ax.set_xlim(time[0], time[-1])
    ymin = min(np.nanmin(l1), np.nanmin(l4))
    ymax = max(np.nanmax(l1), np.nanmax(l4))
    pad = 0.08 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=fig_help.TICK_FS_2COL - 2,
            handlelength=1.5,
            borderaxespad=0.1,
        )
    style_trace_axis(ax)
    return float(np.nanmin([np.nanmin(l1), np.nanmin(l4)])), float(
        np.nanmax([np.nanmax(l1), np.nanmax(l4)])
    )


def add_panel_label(fig, ax, label):
    fig_help.add_panel_label_fig(
        fig,
        ax,
        label,
        dx=-0.03,
        dy=0.01,
        fontsize=fig_help.PANEL_LABEL_FS_2COL,
    )


def shrink_top_pair_axes(axes, shrink=0.86):
    """Narrow the top imported figure13 panels to prevent cross-panel overlap."""
    left_ax, right_ax = axes
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    left_ax.set_position([left_pos.x0, left_pos.y0, left_pos.width * shrink, left_pos.height])
    right_width = right_pos.width * shrink
    right_ax.set_position([
        right_pos.x1 - right_width,
        right_pos.y0,
        right_width,
        right_pos.height,
    ])


def main():
    empirical = load_empirical_trace()
    empirical_window = select_empirical_window(empirical)
    layer_traces = load_layer_traces()
    figure13_data = figure13.prepare_plot_data()
    if not DENSE_ENERGY_WIDTH.exists():
        raise FileNotFoundError(
            f"Missing {DENSE_ENERGY_WIDTH}. Run "
            "data_processing_code/figure_supply_15/compute_dense_energy_width.py first."
        )
    energy_width = pd.read_csv(DENSE_ENERGY_WIDTH)

    fig = plt.figure(figsize=(16.0, 13.8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        3,
        6,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.94,
        bottom=0.055,
        hspace=0.62,
        wspace=0.34,
        height_ratios=[1.15, 1.0, 1.0],
    )

    fig13_axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6])]
    empirical_axes = [fig.add_subplot(gs[1, 2 * i:2 * i + 2]) for i in range(3)]
    layer_axes = [fig.add_subplot(gs[2, 2 * i:2 * i + 2]) for i in range(3)]
    shrink_top_pair_axes(fig13_axes)

    figure13.draw_panel_b(fig13_axes[0], figure13_data)
    figure13.draw_energy_width_panel(fig13_axes[1], energy_width)

    for idx, (key, title) in enumerate(EMPIRICAL_CASES):
        plot_empirical_trace(
            empirical_axes[idx],
            empirical,
            key,
            title,
            empirical_window,
            show_legend=(idx == 0),
        )

    epsilon_to_trace_idx = {
        float(epsilon): idx for idx, epsilon in enumerate(np.asarray(layer_traces["epsilon"], dtype=float))
    }
    layer_ylim = []
    for plot_idx, epsilon in enumerate(EPSILON_PLOT_ORDER):
        layer_ylim.append(
            plot_layer_trace(
                layer_axes[plot_idx],
                layer_traces,
                epsilon_to_trace_idx[float(epsilon)],
                f"epsilon = {epsilon:g}",
                show_legend=(plot_idx == 0),
            )
        )
    layer_ymin = min(ymin for ymin, _ in layer_ylim)
    layer_ymax = max(ymax for _, ymax in layer_ylim)
    layer_pad = 0.08 * (layer_ymax - layer_ymin + 1e-6)
    for ax in layer_axes:
        ax.set_ylim(layer_ymin - layer_pad, layer_ymax + layer_pad)

    fig.text(
        0.52,
        0.635,
        "Empirical P/SP-associated traces under null rewiring",
        ha="center",
        va="center",
        fontsize=fig_help.AXIS_LABEL_FS_2COL,
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.335,
        "Layer-linear model traces under increasing directional asymmetry",
        ha="center",
        va="center",
        fontsize=fig_help.AXIS_LABEL_FS_2COL,
        fontweight="bold",
    )
    #fig.text(0.055, 0.365, "Time", ha="left", va="center", fontsize=fig_help.TICK_FS_2COL)
    #fig.text(0.055, 0.035, "Time", ha="left", va="center", fontsize=fig_help.TICK_FS_2COL)

    add_panel_label(fig, fig13_axes[0], "A")
    add_panel_label(fig, fig13_axes[1], "B")
    add_panel_label(fig, empirical_axes[0], "C")
    add_panel_label(fig, layer_axes[0], "D")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_WIDTH_STATS.parent.mkdir(parents=True, exist_ok=True)
    (
        energy_width.groupby(["layer", "epsilon"])["energy_well_width"]
        .agg(n="count", mean="mean", sem="sem")
        .reset_index()
        .to_csv(OUT_WIDTH_STATS, index=False)
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {LAYER_TRACE_CACHE}")


if __name__ == "__main__":
    main()
