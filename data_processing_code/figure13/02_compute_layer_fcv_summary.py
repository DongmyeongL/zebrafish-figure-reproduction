#!/usr/bin/env python3
"""Compute the shared dense-epsilon, per-run layer FCV for Figure 13 panels B and C."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

import layer_fcv_dense_model as dense


PACK_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
INPUT = DERIVED_ROOT / "figure13" / "layer_asymmetric_epsilon_linear_data.npz"
OUTPUT = DERIVED_ROOT / "figure13" / "layer_fcv_dense_summary.csv"
EPSILONS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_RUNS = 50


def main() -> None:
    with np.load(INPUT) as source:
        params = {key: source[key] for key in source.files}

    rows = []
    n_layers = len(params["layer_sizes"])
    for epsilon in EPSILONS:
        print(f"epsilon {epsilon:.2f}: {N_RUNS} simulations", flush=True)
        for run in range(N_RUNS):
            seed = dense.SEED + int(round(epsilon * 1000)) * 1000 + run
            layer_means = dense.simulate_layer_means_from_original_model(
                params, float(epsilon), seed=seed
            )
            layer_fc_state, layer_fcv = dense.layer_dynamic_fc(
                layer_means, float(params["dt"])
            )
            for layer in range(n_layers):
                values = layer_fc_state[:, layer]
                trimmed = values[
                    (values >= np.nanpercentile(values, 0.5))
                    & (values <= np.nanpercentile(values, 99.5))
                ]
                rows.append(
                    {
                        "epsilon": float(epsilon),
                        "run": run,
                        "layer": f"Layer {layer + 1}",
                        "fc_state_mean": float(np.nanmean(values)),
                        "fc_state_std": float(np.nanstd(values, ddof=1)),
                        "fc_state_width_p95_p05": float(
                            np.nanpercentile(trimmed, 95)
                            - np.nanpercentile(trimmed, 5)
                        ),
                        "fcv_from_fc_state": float(layer_fcv[layer]),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
