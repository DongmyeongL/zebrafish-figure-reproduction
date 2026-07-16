#!/usr/bin/env python3
"""Compute local rank-1 directional coreness and cell DCA by root area."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from common import (
    CONFIG,
    DERIVED_DIR,
    N_REGIONS,
    REGION_NAMES,
    cell_dca_path,
    compact_sc_path,
    ensure_output_dirs,
)


def rank1_dca(matrix: sparse.csr_matrix, max_iter: int, tol: float):
    n = matrix.shape[0]
    c_out = np.full(n, 1.0 / np.sqrt(max(n, 1)), dtype=float)
    c_in = c_out.copy()
    delta = np.nan
    for iteration in range(1, max_iter + 1):
        next_out = matrix @ c_in
        norm = np.linalg.norm(next_out)
        if norm <= 0:
            return c_out * np.nan, c_in * np.nan, iteration, np.nan
        next_out /= norm
        next_in = matrix.T @ next_out
        norm = np.linalg.norm(next_in)
        if norm <= 0:
            return c_out * np.nan, c_in * np.nan, iteration, np.nan
        next_in /= norm
        delta = max(np.linalg.norm(next_out - c_out), np.linalg.norm(next_in - c_in))
        c_out, c_in = next_out, next_in
        if delta < tol:
            break
    return c_out, c_in, iteration, float(delta)


def main() -> None:
    ensure_output_dirs()
    rows = []
    max_iter = int(CONFIG["rank1_max_iter"])
    tol = float(CONFIG["rank1_tolerance"])
    for subject in CONFIG["subjects"]:
        print(f"[Figure 12] subject {subject}: computing local rank-1 DCA", flush=True)
        payload = np.load(compact_sc_path(subject), allow_pickle=False)
        edges = np.asarray(payload["edges"], dtype=np.int64)
        neuron_region = np.asarray(payload["neuron_region"], dtype=np.int64)
        n_neurons = len(neuron_region)
        pre, post = edges[:, 0], edges[:, 1]
        keep = pre != post if CONFIG["exclude_self_edges"] else np.ones(len(edges), dtype=bool)
        graph = sparse.coo_matrix(
            (np.ones(int(keep.sum()), dtype=float), (pre[keep], post[keep])),
            shape=(n_neurons, n_neurons),
        ).tocsr()
        graph.sum_duplicates()
        graph.data[:] = 1.0

        c_out = np.full(n_neurons, np.nan, dtype=float)
        c_in = np.full(n_neurons, np.nan, dtype=float)
        for region_id in range(N_REGIONS):
            idx = np.flatnonzero(neuron_region == region_id)
            local_edges = 0
            iteration = 0
            delta = np.nan
            if idx.size >= 2:
                sub = graph[idx, :][:, idx].tocsr()
                sub.eliminate_zeros()
                local_edges = int(sub.nnz)
                if local_edges:
                    out, inn, iteration, delta = rank1_dca(sub, max_iter, tol)
                    c_out[idx] = out
                    c_in[idx] = inn
            rows.append(
                {
                    "Subject": subject,
                    "RegionID": region_id,
                    "node": REGION_NAMES[region_id],
                    "n_region_neurons": int(idx.size),
                    "n_local_binary_edges": local_edges,
                    "rank1_iterations": int(iteration),
                    "rank1_delta": delta,
                    "n_finite_dca_cells": int(np.isfinite(c_out[idx] - c_in[idx]).sum()),
                }
            )
        np.savez(cell_dca_path(subject), c_out=c_out, c_in=c_in, dca=c_out - c_in)

    output = DERIVED_DIR / "figure12_local_dca_qc.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
