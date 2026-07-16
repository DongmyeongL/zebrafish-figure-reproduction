import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot figures from layer asymmetric epsilon linear-model results."
    )
    parser.add_argument("--data", default="layer_asymmetric_epsilon_linear_data.npz")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def mean_and_sem(values):
    mean = np.nanmean(values, axis=1)
    sem = np.nanstd(values, axis=1) / np.sqrt(values.shape[1])
    return mean, sem


def draw_layer_network_panel(ax, args):
    layer_sizes = list(args.layer_sizes)
    n_layers = len(layer_sizes)
    y_positions = np.arange(n_layers - 1, -1, -1)
    max_size = max(layer_sizes)
    positions = []

    for layer_idx, size in enumerate(layer_sizes):
        x_positions = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        layer_positions = [(x, y_positions[layer_idx]) for x in x_positions]
        positions.append(layer_positions)

    for layer_positions in positions:
        for node_idx in range(len(layer_positions) - 1):
            x0, y0 = layer_positions[node_idx]
            x1, y1 = layer_positions[node_idx + 1]
            ax.plot([x0, x1], [y0, y1], color="0.65", lw=1.1, zorder=1)

    max_scale = max(args.inter_epsilon_scales)
    for layer_idx in range(n_layers - 1):
        scale = args.inter_epsilon_scales[layer_idx]
        upper_center = (max_size / 2.0 + 0.75, y_positions[layer_idx])
        lower_center = (max_size / 2.0 + 0.75, y_positions[layer_idx + 1])

        for source_x, source_y in positions[layer_idx]:
            for target_x, target_y in positions[layer_idx + 1]:
                ax.plot(
                    [source_x + 0.03, target_x + 0.03],
                    [source_y, target_y],
                    color="#d95f02",
                    lw=0.35 + 0.55 * scale / max_scale,
                    alpha=0.07 + 0.12 * scale / max_scale,
                    zorder=0,
                )
                ax.plot(
                    [target_x - 0.03, source_x - 0.03],
                    [target_y, source_y],
                    color="#377eb8",
                    lw=0.35,
                    alpha=0.06,
                    zorder=0,
                )

        forward = FancyArrowPatch(
            upper_center,
            lower_center,
            arrowstyle="-|>",
            mutation_scale=16,
            lw=2.0 + 2.4 * scale / max_scale,
            color="#d95f02",
            alpha=0.85,
            connectionstyle="arc3,rad=-0.35",
            zorder=2,
        )
        backward = FancyArrowPatch(
            (upper_center[0] + 0.28, lower_center[1]),
            (upper_center[0] + 0.28, upper_center[1]),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.0,
            color="#377eb8",
            alpha=0.65,
            connectionstyle="arc3,rad=-0.35",
            zorder=2,
        )
        ax.add_patch(forward)
        ax.add_patch(backward)
        ax.text(
            upper_center[0] + 0.45,
            (upper_center[1] + lower_center[1]) / 2.0,
            f"s={scale:g}",
            ha="left",
            va="center",
            fontsize=8,
            color="#d95f02",
        )

    for layer_idx, layer_positions in enumerate(positions):
        xs, ys = zip(*layer_positions)
        ax.scatter(xs, ys, s=95, color="white", edgecolor="black", linewidth=1.1, zorder=3)
        ax.text(
            -max_size / 2.0 - 0.75,
            y_positions[layer_idx],
            f"L{layer_idx + 1}\nn={layer_sizes[layer_idx]}",
            ha="right",
            va="center",
            fontsize=9,
        )

    ax.text(max_size / 2.0 + 0.35, n_layers - 0.25, "directional coupling", color="0.15", fontsize=8)
    ax.text(max_size / 2.0 + 0.35, n_layers - 0.52, "down: w(1+s epsilon)", color="#d95f02", fontsize=8)
    ax.text(max_size / 2.0 + 0.35, n_layers - 0.79, "up: w(1-s epsilon)", color="#377eb8", fontsize=8)
    ax.text(-max_size / 2.0 - 0.55, n_layers - 0.52, "within layer: symmetric", color="0.35", fontsize=8)
    ax.set_xlim(-max_size / 2.0 - 1.1, max_size / 2.0 + 2.35)
    ax.set_ylim(-0.65, n_layers - 0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("a) Network model")
    ax.set_frame_on(False)


def plot_summary(results, args, output_path):
    eps = results["epsilon_values"]
    mean_fc_mean, mean_fc_sem = mean_and_sem(results["layer_mean_fc"])
    std_mean, std_sem = mean_and_sem(results["layer_temporal_std_fc"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)

    draw_layer_network_panel(axes[0], args)

    ax = axes[1]
    for layer_idx in range(len(args.layer_sizes)):
        label = f"layer {layer_idx + 1} (n={args.layer_sizes[layer_idx]})"
        ax.fill_between(
            eps,
            mean_fc_mean[:, layer_idx] - mean_fc_sem[:, layer_idx],
            mean_fc_mean[:, layer_idx] + mean_fc_sem[:, layer_idx],
            alpha=0.18,
        )
        ax.plot(eps, mean_fc_mean[:, layer_idx], marker="o", label=label)
    ax.set_xlabel("epsilon: between-layer asymmetry")
    ax.set_ylabel("mean FC to other layers")
    ax.set_title("b) Layer-mean FC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    for layer_idx in range(len(args.layer_sizes)):
        label = f"layer {layer_idx + 1} (n={args.layer_sizes[layer_idx]})"
        ax.fill_between(
            eps,
            std_mean[:, layer_idx] - std_sem[:, layer_idx],
            std_mean[:, layer_idx] + std_sem[:, layer_idx],
            alpha=0.18,
        )
        ax.plot(eps, std_mean[:, layer_idx], marker="o", label=label)
    ax.set_xlabel("epsilon: between-layer asymmetry")
    ax.set_ylabel("between-layer temporal std FC")
    ax.set_title("c) Layer-mean std FC")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(
        "4-layer linear network, "
        f"sizes={list(args.layer_sizes)}, intra epsilon={args.intra_epsilon:g}, "
        f"inter scales={list(args.inter_epsilon_scales)}, "
        f"slow drive={args.epsilon_slow_layer_drive:g}"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_layer_fc_heatmap(results, args, output_path, result_key, title, colorbar_label):
    eps = results["epsilon_values"]
    fc_mean = np.nanmean(results[result_key], axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    im = ax.imshow(
        fc_mean,
        origin="lower",
        aspect="auto",
        extent=[0.5, len(args.layer_sizes) + 0.5, eps[0], eps[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("epsilon: between-layer asymmetry")
    ax.set_xticks(np.arange(1, len(args.layer_sizes) + 1))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_representative_traces(results, output_path):
    representative = results.get("representative", {})
    if not representative:
        return False

    fig, axes = plt.subplots(
        len(representative),
        1,
        figsize=(10, 6),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for ax, (epsilon, data) in zip(axes, representative.items()):
        time = data["time"]
        signals = data["signals"]
        ax.plot(time, data["stimulus"], color="black", lw=1.0, label="input")
        for layer_idx, (start, stop) in enumerate(data["slices"]):
            ax.plot(
                time,
                np.mean(signals[:, start:stop], axis=1),
                lw=0.9,
                label=f"layer {layer_idx + 1}",
            )
        ax.set_title(f"epsilon = {epsilon:.2f}")
        ax.set_ylabel("mean signal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    axes[-1].set_xlabel("time")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_network_matrix(results, output_path):
    representative = results.get("representative", {})
    if not representative:
        return False

    epsilon = sorted(representative.keys())[len(representative) // 2]
    data = representative[epsilon]
    j_mat = data["j_mat"]

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    im = ax.imshow(j_mat, cmap="coolwarm")
    for _, stop in data["slices"][:-1]:
        ax.axhline(stop - 0.5, color="black", lw=0.8)
        ax.axvline(stop - 0.5, color="black", lw=0.8)
    ax.set_xlabel("source neuron")
    ax.set_ylabel("target neuron")
    ax.set_title(f"Layered network J, epsilon={epsilon:.2f}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("coupling")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_all_figures(results, args, output_prefix):
    output_paths = {
        "summary": f"{output_prefix}_summary.png",
        "mean_heatmap": f"{output_prefix}_layer_mean_fc_heatmap.png",
        "std_heatmap": f"{output_prefix}_layer_fc_std_heatmap.png",
        "traces": f"{output_prefix}_representative_traces.png",
        "matrix": f"{output_prefix}_matrix.png",
    }

    plot_summary(results, args, output_paths["summary"])
    plot_layer_fc_heatmap(
        results,
        args,
        output_paths["mean_heatmap"],
        "layer_mean_fc",
        "Layer-mean FC",
        "mean FC to other layers",
    )
    plot_layer_fc_heatmap(
        results,
        args,
        output_paths["std_heatmap"],
        "layer_temporal_std_fc",
        "Layer-mean between-layer temporal std FC",
        "mean layer-pair std over time",
    )
    if not plot_representative_traces(results, output_paths["traces"]):
        output_paths.pop("traces")
    if not plot_network_matrix(results, output_paths["matrix"]):
        output_paths.pop("matrix")

    return output_paths


def _scalar(data, key):
    value = data[key]
    if np.ndim(value) == 0:
        return value.item()
    return value


def load_results(data_path):
    data = np.load(data_path, allow_pickle=True)
    results = {
        "epsilon_values": data["epsilon_values"],
        "layer_mean_fc": data["layer_mean_fc"],
        "layer_std_fc": data["layer_std_fc"],
        "layer_temporal_std_fc": data["layer_temporal_std_fc"],
        "layer_gains": data["layer_gains"],
        "terminal_layer_gain": data["terminal_layer_gain"],
        "gain_amplification": data["gain_amplification"],
        "spectral_abscissa": data["spectral_abscissa"],
        "nonnormality": data["nonnormality"],
        "seed": _scalar(data, "seed"),
        "window_size": _scalar(data, "window_size") if "window_size" in data else None,
        "step_size": _scalar(data, "step_size") if "step_size" in data else None,
    }
    if "representative" in data:
        results["representative"] = data["representative"].item()

    args = SimpleNamespace(
        layer_sizes=data["layer_sizes"].astype(int).tolist(),
        n_iter=int(_scalar(data, "n_iter")),
        n_steps=int(_scalar(data, "n_steps")),
        dt=float(_scalar(data, "dt")),
        gamma=float(_scalar(data, "gamma")),
        w_intra=float(_scalar(data, "w_intra")),
        w_inter=float(_scalar(data, "w_inter")),
        intra_epsilon=float(_scalar(data, "intra_epsilon")),
        inter_epsilon_scales=data["inter_epsilon_scales"].astype(float).tolist(),
        layer_decay_offsets=data["layer_decay_offsets"].astype(float).tolist(),
        noise_sigma=float(_scalar(data, "noise_sigma")),
        epsilon_slow_layer_drive=float(_scalar(data, "epsilon_slow_layer_drive")),
        slow_drive_frequency=float(_scalar(data, "slow_drive_frequency")),
        slow_layer_scales=data["slow_layer_scales"].astype(float).tolist(),
        input_amplitude=float(_scalar(data, "input_amplitude")),
        input_frequency=float(_scalar(data, "input_frequency")),
        window_sec=float(_scalar(data, "window_sec")),
        step_sec=float(_scalar(data, "step_sec")),
        seed=_scalar(data, "seed"),
    )
    return results, args


def main():
    args = parse_args()
    results, result_args = load_results(args.data)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.data.removesuffix("_data.npz")
    output_paths = plot_all_figures(results, result_args, output_prefix)
    for output_path in output_paths.values():
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
