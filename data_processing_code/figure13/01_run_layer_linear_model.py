#!/usr/bin/env python3
"""Run the Figure 13 four-layer linear model and save the epsilon sweep."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

from layer_linear_model import run_epsilon_sweep, save_results


PACK_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
OUTPUT = DERIVED_ROOT / "figure13" / "layer_asymmetric_epsilon_linear_data.npz"


def model_parameters() -> SimpleNamespace:
    """Parameters used for the Figure 13 layer-model sweep."""
    return SimpleNamespace(
        layer_sizes=[2, 3, 4, 5],
        n_grid=21,
        n_iter=30,
        n_steps=6000,
        dt=0.005,
        gamma=12.7,
        w_intra=0.25,
        w_inter=1.85,
        intra_epsilon=0.0,
        inter_epsilon_scales=[0.8, 0.6, 0.1],
        noise_sigma=0.15,
        epsilon_slow_layer_drive=1.0,
        slow_drive_frequency=0.1,
        slow_layer_scales=[1.0, 1.0, 1.0, 1.0],
        layer_decay_offsets=[0.0, 0.0, 0.0, 0.0],
        input_amplitude=0.0,
        input_frequency=0.85,
        window_sec=3.0,
        step_sec=1.5,
        seed=3526228248,
    )


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    args = model_parameters()
    results = run_epsilon_sweep(args)
    save_results(results, args, OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
