import argparse

import numpy as np

from plot_layer_asymmetric_epsilon_linear_model import plot_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a 4-layer directed linear-network epsilon sweep."
    )
    parser.add_argument("--layer-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--n-grid", type=int, default=21)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=6000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--gamma",
        type=float,
        default=12.7,
        help="Diagonal decay strength. Larger values make the linear system more stable.",
    )
    parser.add_argument(
        "--w-intra",
        type=float,
        default=0.25,
        help="Base nearest-neighbor coupling strength inside each layer.",
    )
    parser.add_argument(
        "--w-inter",
        type=float,
        default=1.85,
        help="Base all-to-all coupling strength between adjacent layers.",
    )
    parser.add_argument(
        "--intra-epsilon",
        type=float,
        default=0.0,
        help="Fixed forward/backward asymmetry inside each layer. Default 0.0 is symmetric.",
    )
    parser.add_argument(
        "--inter-epsilon-scales",
        type=float,
        nargs="+",
        default=[0.8, 0.6, 0.1],
        help="Per-gap scaling for between-layer epsilon. Default: L1-L2=0.8, L2-L3=0.6, L3-L4=0.1.",
    )
    parser.add_argument("--noise-sigma", type=float, default=0.15)
    parser.add_argument(
        "--epsilon-slow-layer-drive",
        type=float,
        default=1.0,
        help="Amplitude of slow layer-level fluctuations.",
    )
    parser.add_argument(
        "--slow-drive-frequency",
        type=float,
        default=0.1,
        help="Base frequency of slow layer fluctuations.",
    )
    parser.add_argument(
        "--slow-layer-scales",
        type=float,
        nargs="+",
        default=[1.0, 1.0, 1.0, 1.0],
        help="Layer-wise scale for slow fluctuations.",
    )
    parser.add_argument(
        "--layer-decay-offsets",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0, 0.0],
        help="Extra diagonal decay per layer. Default treats L4 as a stabilized output layer.",
    )
    parser.add_argument("--input-amplitude", type=float, default=0.0)
    parser.add_argument("--input-frequency", type=float, default=0.85)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", default="layer_asymmetric_epsilon_linear")
    return parser.parse_args()


def layer_slices(layer_sizes):
    slices = []
    start = 0
    for size in layer_sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return slices


def build_layer_jacobian(
    layer_sizes,
    gamma,
    w_intra,
    w_inter,
    intra_epsilon,
    epsilon,
    inter_epsilon_scales,
    layer_decay_offsets,
):
    """Build J for dx/dt = J x on a directed layered network."""
    n_nodes = sum(layer_sizes)
    j_mat = -gamma * np.eye(n_nodes, dtype=float)
    slices = layer_slices(layer_sizes)

    intra_forward = w_intra * (1.0 + intra_epsilon)
    intra_backward = w_intra * (1.0 - intra_epsilon)
    for layer_idx, layer_slice in enumerate(slices):
        if layer_decay_offsets[layer_idx] != 0.0:
            j_mat[layer_slice, layer_slice] -= layer_decay_offsets[layer_idx] * np.eye(
                layer_slice.stop - layer_slice.start
            )
        nodes = np.arange(layer_slice.start, layer_slice.stop)
        for local_idx in range(len(nodes) - 1):
            source = nodes[local_idx]
            target = nodes[local_idx + 1]
            j_mat[target, source] = intra_forward
            if intra_backward > 0.0:
                j_mat[source, target] = intra_backward

    for layer_idx in range(len(slices) - 1):
        scaled_epsilon = inter_epsilon_scales[layer_idx] * epsilon
        inter_forward = w_inter * (1.0 + scaled_epsilon)
        inter_backward = w_inter * (1.0 - scaled_epsilon)
        upper_nodes = np.arange(slices[layer_idx].start, slices[layer_idx].stop)
        lower_nodes = np.arange(slices[layer_idx + 1].start, slices[layer_idx + 1].stop)
        j_mat[np.ix_(lower_nodes, upper_nodes)] = inter_forward
        if inter_backward > 0.0:
            j_mat[np.ix_(upper_nodes, lower_nodes)] = inter_backward

    return j_mat, slices


def simulate_layer_network(
    epsilon,
    layer_sizes,
    n_steps,
    dt,
    gamma,
    w_intra,
    w_inter,
    intra_epsilon,
    inter_epsilon_scales,
    layer_decay_offsets,
    noise_sigma,
    epsilon_slow_layer_drive,
    slow_drive_frequency,
    slow_layer_scales,
    input_amplitude,
    input_frequency,
    seed,
):
    rng = np.random.default_rng(seed)
    j_mat, slices = build_layer_jacobian(
        layer_sizes,
        gamma,
        w_intra,
        w_inter,
        intra_epsilon,
        epsilon,
        inter_epsilon_scales,
        layer_decay_offsets,
    )
    n_nodes = sum(layer_sizes)
    state = np.zeros(n_nodes, dtype=float)
    noise_scale = noise_sigma * np.sqrt(dt)
    time = np.arange(n_steps) * dt
    stimulus = np.sin(2.0 * np.pi * input_frequency * time)
    signals = np.empty((n_steps, n_nodes), dtype=float)
    slow_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(slices))
    slow_frequencies = np.full(len(slices), slow_drive_frequency, dtype=float)
    slow_layer_scales = np.asarray(slow_layer_scales, dtype=float)

    input_nodes = np.arange(slices[0].start, slices[0].stop)
    for step in range(n_steps):
        external_drive = np.zeros(n_nodes, dtype=float)
        external_drive[input_nodes] = input_amplitude * stimulus[step]
        if epsilon_slow_layer_drive != 0.0:
            for layer_idx, layer_slice in enumerate(slices):
                slow_drive = (
                    epsilon_slow_layer_drive
                    * slow_layer_scales[layer_idx]
                    * np.sin(
                        2.0 * np.pi * slow_frequencies[layer_idx] * time[step]
                        + slow_phases[layer_idx]
                    )
                )
                external_drive[layer_slice] += slow_drive
        state += dt * (j_mat @ state + external_drive)
        state += noise_scale * rng.standard_normal(n_nodes)
        signals[step] = state

    return time, stimulus, signals, j_mat, slices


def sliding_window_layer_fc_stats(signals, slices, window_size, step_size):
    n_steps = signals.shape[0]
    n_windows = (n_steps - window_size) // step_size + 1
    n_layers = len(slices)
    layer_mean_fc = np.full((n_windows, len(slices)), np.nan)
    layer_to_others_fc = [
        np.full((n_windows, len(slices) - 1), np.nan) for _ in range(len(slices))
    ]
    layer_signals = np.column_stack(
        [np.mean(signals[:, layer_slice], axis=1) for layer_slice in slices]
    )

    for idx in range(n_windows):
        start = idx * step_size
        end = start + window_size
        corr = np.corrcoef(layer_signals[start:end].T)
        corr[~np.isfinite(corr)] = np.nan
        for layer_idx in range(n_layers):
            other_layer_fc = np.delete(corr[layer_idx], layer_idx)
            layer_to_others_fc[layer_idx][idx, :] = other_layer_fc
            layer_mean_fc[idx, layer_idx] = np.nanmean(other_layer_fc)

    layer_mean = np.nanmean(layer_mean_fc, axis=0)
    layer_std = np.asarray([np.nanstd(values) for values in layer_to_others_fc])
    layer_temporal_std = np.asarray(
        [np.nanmean(np.nanstd(values, axis=0)) for values in layer_to_others_fc]
    )
    return layer_mean, layer_std, layer_temporal_std


def stimulus_projection_gain(stimulus, signals):
    centered_stimulus = stimulus - np.mean(stimulus)
    stimulus_power = np.mean(centered_stimulus**2)
    if stimulus_power == 0.0:
        return np.full(signals.shape[1], np.nan)

    centered_signals = signals - np.mean(signals, axis=0, keepdims=True)
    projection = np.mean(centered_signals * centered_stimulus[:, None], axis=0)
    return np.abs(projection) / stimulus_power


def summarize_trial(stimulus, signals, j_mat, slices, window_size, step_size):
    layer_mean_fc, layer_std_fc, layer_temporal_std_fc = sliding_window_layer_fc_stats(
        signals,
        slices,
        window_size,
        step_size,
    )
    gains = stimulus_projection_gain(stimulus, signals)
    layer_gains = np.asarray([np.nanmean(gains[layer_slice]) for layer_slice in slices])
    eigvals = np.linalg.eigvals(j_mat)

    return {
        "layer_mean_fc": layer_mean_fc,
        "layer_std_fc": layer_std_fc,
        "layer_temporal_std_fc": layer_temporal_std_fc,
        "layer_gains": layer_gains,
        "terminal_layer_gain": layer_gains[-1],
        "gain_amplification": layer_gains[-1] / max(layer_gains[0], 1e-12),
        "spectral_abscissa": np.max(np.real(eigvals)),
        "nonnormality": np.linalg.norm(j_mat.T @ j_mat - j_mat @ j_mat.T, ord="fro"),
    }


def run_epsilon_sweep(args):
    if len(args.layer_sizes) != 4:
        raise ValueError("This script expects exactly 4 layers.")
    if len(args.inter_epsilon_scales) != len(args.layer_sizes) - 1:
        raise ValueError(
            "--inter-epsilon-scales must have one value per adjacent layer gap."
        )
    if len(args.slow_layer_scales) != len(args.layer_sizes):
        raise ValueError("--slow-layer-scales must have one value per layer.")
    if len(args.layer_decay_offsets) != len(args.layer_sizes):
        raise ValueError("--layer-decay-offsets must have one value per layer.")
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

    epsilon_values = np.linspace(0.0, 1.0, args.n_grid)
    window_size = int(round(args.window_sec / args.dt))
    step_size = int(round(args.step_sec / args.dt))
    seed_seq = np.random.SeedSequence(args.seed)
    grid_seed_seq = seed_seq.spawn(len(epsilon_values))

    n_layers = len(args.layer_sizes)
    layer_mean_fc = np.full((len(epsilon_values), args.n_iter, n_layers), np.nan)
    layer_std_fc = np.full_like(layer_mean_fc, np.nan)
    layer_temporal_std_fc = np.full_like(layer_mean_fc, np.nan)
    layer_gains = np.full_like(layer_mean_fc, np.nan)
    terminal_layer_gain = np.full((len(epsilon_values), args.n_iter), np.nan)
    gain_amplification = np.full_like(terminal_layer_gain, np.nan)
    spectral_abscissa = np.full(len(epsilon_values), np.nan)
    nonnormality = np.full(len(epsilon_values), np.nan)
    representative = {}

    for eps_idx, epsilon in enumerate(epsilon_values):
        print(f"epsilon {eps_idx + 1}/{len(epsilon_values)}: {epsilon:.3f}")
        trial_seeds = grid_seed_seq[eps_idx].generate_state(args.n_iter, dtype=np.uint32)

        for trial_idx, trial_seed in enumerate(trial_seeds):
            time, stimulus, signals, j_mat, slices = simulate_layer_network(
                epsilon=epsilon,
                layer_sizes=args.layer_sizes,
                n_steps=args.n_steps,
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
                seed=int(trial_seed),
            )
            summary = summarize_trial(stimulus, signals, j_mat, slices, window_size, step_size)
            layer_mean_fc[eps_idx, trial_idx] = summary["layer_mean_fc"]
            layer_std_fc[eps_idx, trial_idx] = summary["layer_std_fc"]
            layer_temporal_std_fc[eps_idx, trial_idx] = summary["layer_temporal_std_fc"]
            layer_gains[eps_idx, trial_idx] = summary["layer_gains"]
            terminal_layer_gain[eps_idx, trial_idx] = summary["terminal_layer_gain"]
            gain_amplification[eps_idx, trial_idx] = summary["gain_amplification"]
            spectral_abscissa[eps_idx] = summary["spectral_abscissa"]
            nonnormality[eps_idx] = summary["nonnormality"]

            if trial_idx == 0 and eps_idx in {0, len(epsilon_values) // 2, len(epsilon_values) - 1}:
                representative[float(epsilon)] = {
                    "time": time,
                    "stimulus": stimulus,
                    "signals": signals,
                    "j_mat": j_mat,
                    "slices": [(layer_slice.start, layer_slice.stop) for layer_slice in slices],
                }

    return {
        "epsilon_values": epsilon_values,
        "layer_mean_fc": layer_mean_fc,
        "layer_std_fc": layer_std_fc,
        "layer_temporal_std_fc": layer_temporal_std_fc,
        "layer_gains": layer_gains,
        "terminal_layer_gain": terminal_layer_gain,
        "gain_amplification": gain_amplification,
        "spectral_abscissa": spectral_abscissa,
        "nonnormality": nonnormality,
        "representative": representative,
        "seed": args.seed,
        "window_size": window_size,
        "step_size": step_size,
    }


def save_results(results, args, output_path):
    np.savez(
        output_path,
        epsilon_values=results["epsilon_values"],
        layer_mean_fc=results["layer_mean_fc"],
        layer_std_fc=results["layer_std_fc"],
        layer_temporal_std_fc=results["layer_temporal_std_fc"],
        layer_gains=results["layer_gains"],
        terminal_layer_gain=results["terminal_layer_gain"],
        gain_amplification=results["gain_amplification"],
        spectral_abscissa=results["spectral_abscissa"],
        nonnormality=results["nonnormality"],
        representative=np.asarray(results["representative"], dtype=object),
        window_size=results["window_size"],
        step_size=results["step_size"],
        layer_sizes=np.asarray(args.layer_sizes),
        n_iter=args.n_iter,
        n_steps=args.n_steps,
        dt=args.dt,
        gamma=args.gamma,
        w_intra=args.w_intra,
        w_inter=args.w_inter,
        intra_epsilon=args.intra_epsilon,
        inter_epsilon_scales=np.asarray(args.inter_epsilon_scales),
        layer_decay_offsets=np.asarray(args.layer_decay_offsets),
        noise_sigma=args.noise_sigma,
        epsilon_slow_layer_drive=args.epsilon_slow_layer_drive,
        slow_drive_frequency=args.slow_drive_frequency,
        slow_layer_scales=np.asarray(args.slow_layer_scales),
        input_amplitude=args.input_amplitude,
        input_frequency=args.input_frequency,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        seed=results["seed"],
    )


def main():
    args = parse_args()
    results = run_epsilon_sweep(args)
    data_path = f"{args.output_prefix}_data.npz"

    save_results(results, args, data_path)
    output_paths = plot_all_figures(results, args, args.output_prefix)

    print(f"Saved {data_path}")
    for output_path in output_paths.values():
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
