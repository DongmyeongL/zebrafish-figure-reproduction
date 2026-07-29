#!/usr/bin/env python3
"""Compute cell-level local and functional-unit interregional rank-1 DCA."""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from scipy import sparse

from common import (
    CONFIG,
    DERIVED_DIR,
    N_REGIONS,
    REGION_NAMES,
    SC_SOURCE_CHOICES,
    cell_dca_path,
    compact_sc_path,
    ensure_output_dirs,
    functional_unit_dca_path,
    original_sc_path,
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


def functional_unit_membership(subject: int, neuron_region: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each SC neuron to its saved Figure-9 spatial functional unit."""
    with original_sc_path(subject).open("rb") as handle:
        raw = pickle.load(handle)
    membership = np.full(len(neuron_region), -1, dtype=np.int64)
    unit_region = np.asarray(raw["root_area"], dtype=np.int64)
    unit_size = np.zeros(len(unit_region), dtype=np.int64)
    for unit_id, cells in enumerate(raw["final_id_cluster"]):
        cells = np.asarray(cells, dtype=np.int64)
        membership[cells] = unit_id
        unit_size[unit_id] = len(cells)
    if (membership < 0).any():
        raise ValueError(f"Subject {subject}: some SC neurons are not assigned to a functional unit")
    expected_region = unit_region[membership]
    if not np.array_equal(expected_region, neuron_region):
        raise ValueError(
            f"Subject {subject}: compact SC neuron order does not match saved functional units"
        )
    return membership, unit_region, unit_size


def compute_functional_unit_dca(
    subject: int,
    edges: np.ndarray,
    neuron_region: np.ndarray,
    max_iter: int,
    tol: float,
    sc_source: str,
) -> list[dict]:
    """Apply rank-1 power iteration to the functional-unit interregional SC."""
    membership, unit_region, unit_size = functional_unit_membership(subject, neuron_region)
    pre, post = edges[:, 0], edges[:, 1]
    source_unit, target_unit = membership[pre], membership[post]
    inter = (source_unit != target_unit) & (unit_region[source_unit] != unit_region[target_unit])
    n_units = len(unit_region)
    graph = sparse.coo_matrix(
        (np.ones(int(inter.sum()), dtype=float), (source_unit[inter], target_unit[inter])),
        shape=(n_units, n_units),
    ).tocsr()
    graph.sum_duplicates()
    c_out, c_in, iteration, delta = rank1_dca(graph, max_iter, tol)
    dca = c_out - c_in
    out_strength = np.asarray(graph.sum(axis=1)).ravel()
    in_strength = np.asarray(graph.sum(axis=0)).ravel()
    np.savez(
        functional_unit_dca_path(subject, sc_source),
        c_out=c_out,
        c_in=c_in,
        dca=dca,
        unit_region=unit_region,
        unit_size=unit_size,
        inter_out_strength=out_strength,
        inter_in_strength=in_strength,
        rank1_iterations=np.asarray(iteration),
        rank1_delta=np.asarray(delta),
    )
    return [
        {
            "Subject": subject,
            "SC_source": sc_source,
            "FunctionalUnit": unit_id,
            "RegionID": int(unit_region[unit_id]),
            "node": REGION_NAMES[unit_region[unit_id]],
            "n_unit_neurons": int(unit_size[unit_id]),
            "inter_out_strength": int(out_strength[unit_id]),
            "inter_in_strength": int(in_strength[unit_id]),
            "c_out": c_out[unit_id],
            "c_in": c_in[unit_id],
            "FU_DCA": dca[unit_id],
            "rank1_iterations": iteration,
            "rank1_delta": delta,
        }
        for unit_id in range(n_units)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sc-source",
        choices=SC_SOURCE_CHOICES,
        default="fcs_calibrated_skeleton_kmeans_nearest_r12",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=CONFIG["subjects"])
    parser.add_argument("--skip-cell-dca", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    cell_rows = []
    functional_unit_rows = []
    max_iter = int(CONFIG["rank1_max_iter"])
    tol = float(CONFIG["rank1_tolerance"])
    for subject in args.subjects:
        print(f"[Figure 12] subject {subject}: loading {args.sc_source} SC", flush=True)
        payload = np.load(compact_sc_path(subject, args.sc_source), allow_pickle=False)
        edges = np.asarray(payload["edges"], dtype=np.int64)
        neuron_region = np.asarray(payload["neuron_region"], dtype=np.int64)
        functional_unit_rows.extend(
            compute_functional_unit_dca(
                subject, edges, neuron_region, max_iter, tol, args.sc_source
            )
        )
        if args.skip_cell_dca:
            continue
        print(f"[Figure 12] subject {subject}: computing local rank-1 cell DCA", flush=True)
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
            cell_rows.append(
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
        np.savez(
            cell_dca_path(subject, args.sc_source),
            c_out=c_out,
            c_in=c_in,
            dca=c_out - c_in,
        )

    source_dir = DERIVED_DIR / "functional_unit_dca" / args.sc_source
    source_dir.mkdir(parents=True, exist_ok=True)
    fu_output = source_dir / "figure12_functional_unit_dca_qc.csv"
    fu_current = pd.DataFrame(functional_unit_rows)
    if fu_output.exists():
        fu_current = pd.concat((pd.read_csv(fu_output), fu_current), ignore_index=True)
        fu_current = fu_current.drop_duplicates(
            subset=("Subject", "SC_source", "FunctionalUnit"), keep="last"
        ).sort_values(["Subject", "FunctionalUnit"])
    fu_current.to_csv(fu_output, index=False)
    print(f"Saved {fu_output}")
    if cell_rows:
        output = DERIVED_DIR / "figure12_local_dca_qc.csv"
        if args.sc_source != "historical":
            output = DERIVED_DIR / "cell_dca" / args.sc_source / "figure12_local_dca_qc.csv"
        pd.DataFrame(cell_rows).to_csv(output, index=False)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
