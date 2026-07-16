#!/usr/bin/env python3
"""Reduce pair-level Supply 1 inputs to the compact public plotting input."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import lognorm


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-data",
        type=Path,
        required=True,
        help="NPZ containing dist_data and positive fc_data arrays.",
    )
    parser.add_argument(
        "--soma-distance-data",
        type=Path,
        required=True,
        help="NPY containing soma-to-endpoint distances.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "raw_data" / "figure_supply_1" / "figure_supply_1_compact.npz",
    )
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--max-distance", type=float, default=500.0)
    args = parser.parse_args()

    source = np.load(args.pair_data)
    distance = np.asarray(source["dist_data"], dtype=float)
    fc = np.asarray(source["fc_data"], dtype=float)
    valid = np.isfinite(distance) & np.isfinite(fc) & (fc > 0)
    distance, fc = distance[valid], fc[valid]

    edges = np.linspace(float(distance.min()), args.max_distance, args.bins + 1)
    membership = np.digitize(distance, edges)
    d, shape, scale = [], [], []
    for index in range(1, args.bins + 1):
        values = fc[membership == index]
        if values.size == 0:
            continue
        fitted_shape, _, fitted_scale = lognorm.fit(values, floc=0)
        d.append(edges[index])
        shape.append(fitted_shape)
        scale.append(fitted_scale)

    soma_distance = np.asarray(np.load(args.soma_distance_data), dtype=float)
    hist_counts, hist_edges = np.histogram(soma_distance[np.isfinite(soma_distance)], bins=args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        d=np.asarray(d),
        lognorm_shape=np.asarray(shape),
        lognorm_scale=np.asarray(scale),
        hist_counts=hist_counts,
        hist_edges=hist_edges,
        n_synapse_dist=int(np.isfinite(soma_distance).sum()),
        n_fc_pairs=int(valid.sum()),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
