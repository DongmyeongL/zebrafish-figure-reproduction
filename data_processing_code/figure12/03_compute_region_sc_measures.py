#!/usr/bin/env python3
"""Compute all five Figure 12 structural measures from the same directed SC."""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import random

import igraph as ig
import numpy as np
import pandas as pd

from common import (
    CONFIG,
    DERIVED_DIR,
    N_REGIONS,
    REGION_NAMES,
    cell_dca_path,
    compact_sc_path,
    ensure_output_dirs,
)


def aggregate_mean(group, weights, n=N_REGIONS):
    valid = np.isfinite(weights)
    numer = np.bincount(group[valid], weights=weights[valid], minlength=n)
    denom = np.bincount(group[valid], minlength=n)
    return np.divide(numer, denom, out=np.full(n, np.nan), where=denom > 0), denom


def within_region_modularity(edges, neuron_region, region_id):
    nodes = np.flatnonzero(neuron_region == region_id)
    if nodes.size < 2:
        return np.nan, 0, 0
    membership = np.full(len(neuron_region), -1, dtype=np.int64)
    membership[nodes] = np.arange(nodes.size)
    pre, post = edges[:, 0], edges[:, 1]
    keep = (
        (neuron_region[pre] == region_id)
        & (neuron_region[post] == region_id)
        & (pre != post)
    )
    u = membership[pre[keep]]
    v = membership[post[keep]]
    if len(u) == 0:
        return np.nan, 0, 0
    a, b = np.minimum(u, v), np.maximum(u, v)
    pairs = np.empty(len(a), dtype=[("a", np.int64), ("b", np.int64)])
    pairs["a"], pairs["b"] = a, b
    unique, inverse = np.unique(pairs, return_inverse=True)
    weights = np.bincount(inverse).astype(float)
    graph = ig.Graph(n=int(nodes.size), edges=list(zip(unique["a"], unique["b"])), directed=False)
    graph.es["weight"] = weights.tolist()
    try:
        clustering = graph.community_leiden(objective_function="modularity", weights="weight")
        q = float(graph.modularity(clustering.membership, weights="weight"))
        n_modules = int(len(set(clustering.membership)))
    except Exception:
        q, n_modules = np.nan, 0
    return q, int(len(unique)), n_modules


def main() -> None:
    ensure_output_dirs()
    ig.set_random_number_generator(random.Random(int(CONFIG["random_seed"])))
    rows = []
    offset = float(CONFIG["log_out_in_offset"])
    for subject in CONFIG["subjects"]:
        print(f"[Figure 12] subject {subject}: computing regional SC measures", flush=True)
        sc = np.load(compact_sc_path(subject), allow_pickle=False)
        edges = np.asarray(sc["edges"], dtype=np.int64)
        neuron_region = np.asarray(sc["neuron_region"], dtype=np.int64)
        dca = np.asarray(np.load(cell_dca_path(subject), allow_pickle=False)["dca"], dtype=float)
        pre, post = edges[:, 0], edges[:, 1]
        keep = pre != post if CONFIG["exclude_self_edges"] else np.ones(len(edges), dtype=bool)
        pre, post = pre[keep], post[keep]
        src_region = neuron_region[pre]
        tgt_region = neuron_region[post]
        inter = src_region != tgt_region
        src, tgt = pre[inter], post[inter]
        src_r, tgt_r = src_region[inter], tgt_region[inter]

        post_dca, post_n = aggregate_mean(src_r, dca[tgt])
        pre_dca, pre_n = aggregate_mean(tgt_r, dca[src])
        out_count = np.bincount(src_r, minlength=N_REGIONS)
        in_count = np.bincount(tgt_r, minlength=N_REGIONS)
        log_out_in = np.log((out_count + offset) / (in_count + offset))
        finite_pair = np.isfinite(dca[src]) & np.isfinite(dca[tgt])
        oo = finite_pair & (dca[src] > 0) & (dca[tgt] > 0)
        oo_count = np.bincount(src_r[oo], minlength=N_REGIONS)
        oo_fraction = np.divide(
            oo_count,
            out_count,
            out=np.full(N_REGIONS, np.nan),
            where=out_count > 0,
        )

        for region_id, node in enumerate(REGION_NAMES):
            modularity, modularity_edges, n_modules = within_region_modularity(
                edges, neuron_region, region_id
            )
            rows.append(
                {
                    "Subject": subject,
                    "recording_id": f"subject_{subject}",
                    "RegionID": region_id,
                    "node": node,
                    "PostDCA": post_dca[region_id],
                    "PreDCA": pre_dca[region_id],
                    "OO_fraction": oo_fraction[region_id],
                    "OO_count": int(oo_count[region_id]),
                    "Modularity": modularity,
                    "LogOutIn": log_out_in[region_id],
                    "inter_out_edges": int(out_count[region_id]),
                    "inter_in_edges": int(in_count[region_id]),
                    "post_dca_edges": int(post_n[region_id]),
                    "pre_dca_edges": int(pre_n[region_id]),
                    "modularity_edges": modularity_edges,
                    "modularity_n_modules": n_modules,
                }
            )

    output = DERIVED_DIR / "figure12_subject_region_structural_measures_all72.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
