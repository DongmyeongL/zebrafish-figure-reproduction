#!/usr/bin/env python3
"""Compute energy-well width for all layers on a dense epsilon grid.

The dynamic FC state and potential estimator are identical to the Figure 13
layer-model pipeline. For each epsilon, shared histogram edges are estimated
from all 50 runs within a layer. Each run's potential is then computed and its
contiguous well width around U_min is measured at U <= U_min + 2.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd


PACK_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
FIGURE13_CODE = PACK_ROOT / "data_processing_code" / "figure13"
if str(FIGURE13_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE13_CODE))

import layer_fcv_dense_model as dense


INPUT = DERIVED_ROOT / "figure13" / "layer_asymmetric_epsilon_linear_data.npz"
OUTPUT = DERIVED_ROOT / "figure13" / "layer_energy_width_dense.csv"
EPSILONS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
N_RUNS = 50
DELTA_U = 2.0


def well_width(z: np.ndarray, potential: np.ndarray, delta_u: float = DELTA_U) -> float:
    finite = np.isfinite(z) & np.isfinite(potential)
    z = np.asarray(z, dtype=float)[finite]
    potential = np.asarray(potential, dtype=float)[finite]
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


def output_is_complete() -> bool:
    if not OUTPUT.exists():
        return False
    try:
        table = pd.read_csv(OUTPUT)
    except Exception:
        return False
    expected = len(EPSILONS) * N_RUNS * 4
    counts = table.groupby(["epsilon", "layer"])["run"].nunique()
    return bool(
        len(table) == expected
        and len(counts) == len(EPSILONS) * 4
        and (counts == N_RUNS).all()
        and np.allclose(sorted(table["epsilon"].unique()), EPSILONS)
    )


def main() -> None:
    if output_is_complete():
        print(f"Using complete cached table: {OUTPUT}")
        return

    with np.load(INPUT) as source:
        params = {key: source[key] for key in source.files}
    n_layers = len(params["layer_sizes"])
    rows = []

    for epsilon in EPSILONS:
        print(f"epsilon {epsilon:.2f}: {N_RUNS} simulations", flush=True)
        run_states = []
        for run in range(N_RUNS):
            seed = dense.SEED + int(round(float(epsilon) * 1000)) * 1000 + run
            layer_means = dense.simulate_layer_means_from_original_model(
                params, float(epsilon), seed=seed
            )
            layer_fc_state, _ = dense.layer_dynamic_fc(layer_means, float(params["dt"]))
            run_states.append(layer_fc_state)

        for layer_index in range(n_layers):
            layer_values = [state[:, layer_index] for state in run_states]
            common_edges = dense.robust_edges(np.concatenate(layer_values))
            for run, values in enumerate(layer_values):
                centers, potential = dense.potential_curve(values, common_edges)
                rows.append(
                    {
                        "epsilon": float(epsilon),
                        "run": run,
                        "layer": f"Layer {layer_index + 1}",
                        "energy_well_width": well_width(centers, potential),
                        "delta_u": DELTA_U,
                        "n_dynamic_fc_windows": len(values),
                    }
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
