#!/usr/bin/env python3
"""Compute all five Figure 12 structural measures from the same directed SC."""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pickle
import random

try:
    import igraph as ig
except ImportError:
    ig = None
    import networkx as nx
import numpy as np
import pandas as pd

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


def main_cell() -> None:
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


def functional_unit_membership(subject, neuron_region):
    """Return the saved functional-unit assignment in compact-SC neuron order."""
    with original_sc_path(subject).open("rb") as handle:
        raw = pickle.load(handle)
    membership = np.full(len(neuron_region), -1, dtype=np.int64)
    unit_region = np.asarray(raw["root_area"], dtype=np.int64)
    unit_size = np.zeros(len(unit_region), dtype=np.int64)
    for unit_id, cells in enumerate(raw["final_id_cluster"]):
        cells = np.asarray(cells, dtype=np.int64)
        membership[cells] = unit_id
        unit_size[unit_id] = len(cells)
    if (membership < 0).any() or not np.array_equal(unit_region[membership], neuron_region):
        raise ValueError(f"Subject {subject}: functional-unit membership does not match compact SC order")
    return membership, unit_region, unit_size


def unit_adjacency(edges, membership, unit_region):
    """Aggregate cell edges into the directed, interregional unit adjacency."""
    source_unit = membership[edges[:, 0]]
    target_unit = membership[edges[:, 1]]
    keep = (source_unit != target_unit) & (unit_region[source_unit] != unit_region[target_unit])
    n_units = len(unit_region)
    adjacency = np.zeros((n_units, n_units), dtype=float)
    np.add.at(adjacency, (source_unit[keep], target_unit[keep]), 1.0)
    return adjacency


def global_unit_communities(adjacency):
    """Detect communities on the symmetrized functional-unit interregional graph."""
    symmetric = adjacency + adjacency.T
    source, target = np.nonzero(np.triu(symmetric, 1))
    if len(source) == 0:
        return np.full(len(adjacency), -1, dtype=int), np.nan, 0
    edge_list = list(zip(source.tolist(), target.tolist()))
    weights = symmetric[source, target].tolist()
    if ig is not None:
        graph = ig.Graph(n=len(adjacency), edges=edge_list, directed=False)
        graph.es["weight"] = weights
        clustering = graph.community_leiden(objective_function="modularity", weights="weight")
        membership = np.asarray(clustering.membership, dtype=int)
        modularity = float(graph.modularity(membership.tolist(), weights="weight"))
        return membership, modularity, int(len(set(membership)))

    # Portable fallback for environments without python-igraph.
    graph = nx.Graph()
    graph.add_nodes_from(range(len(adjacency)))
    graph.add_weighted_edges_from((u, v, weight) for (u, v), weight in zip(edge_list, weights))
    communities = list(nx.community.louvain_communities(
        graph, weight="weight", seed=int(CONFIG["random_seed"])
    ))
    membership = np.full(len(adjacency), -1, dtype=int)
    for community_id, nodes in enumerate(communities):
        membership[list(nodes)] = community_id
    modularity = float(nx.community.modularity(graph, communities, weight="weight"))
    return membership, modularity, int(len(communities))


def main_functional_unit(sc_source, subjects):
    """Compute root-area measures from rank-1 functional-unit DCA and unit SC."""
    ensure_output_dirs()
    if ig is not None:
        ig.set_random_number_generator(random.Random(int(CONFIG["random_seed"])))
    rows = []
    community_rows = []
    offset = float(CONFIG["log_out_in_offset"])
    for subject in subjects:
        print(f"[Figure 12] subject {subject}: aggregating functional-unit structural measures", flush=True)
        sc = np.load(compact_sc_path(subject, sc_source), allow_pickle=False)
        edges = np.asarray(sc["edges"], dtype=np.int64)
        neuron_region = np.asarray(sc["neuron_region"], dtype=np.int64)
        membership, unit_region, unit_size = functional_unit_membership(subject, neuron_region)
        unit_data = np.load(functional_unit_dca_path(subject, sc_source), allow_pickle=False)
        dca = np.asarray(unit_data["dca"], dtype=float)
        if len(dca) != len(unit_region):
            raise ValueError(f"Subject {subject}: FU-DCA vector has incompatible length")
        adjacency = unit_adjacency(edges, membership, unit_region)
        out_strength = adjacency.sum(axis=1)
        in_strength = adjacency.sum(axis=0)

        # Existing DCApost/pre and hard OO definitions, evaluated on units.
        post_numerator = adjacency @ dca
        pre_numerator = adjacency.T @ dca
        positive = dca > 0
        hard_oo_numerator = (adjacency * np.outer(positive, positive)).sum(axis=1)
        post_region_num = np.bincount(unit_region, weights=post_numerator, minlength=N_REGIONS)
        pre_region_num = np.bincount(unit_region, weights=pre_numerator, minlength=N_REGIONS)
        oo_region_num = np.bincount(unit_region, weights=hard_oo_numerator, minlength=N_REGIONS)
        out_region = np.bincount(unit_region, weights=out_strength, minlength=N_REGIONS)
        in_region = np.bincount(unit_region, weights=in_strength, minlength=N_REGIONS)
        post_dca = np.divide(post_region_num, out_region, out=np.full(N_REGIONS, np.nan), where=out_region > 0)
        pre_dca = np.divide(pre_region_num, in_region, out=np.full(N_REGIONS, np.nan), where=in_region > 0)
        hard_oo = np.divide(oo_region_num, out_region, out=np.full(N_REGIONS, np.nan), where=out_region > 0)
        log_out_in = np.log((out_region + offset) / (in_region + offset))

        # Directed participation: community-spanning breadth of outgoing flow.
        community, global_q, n_communities = global_unit_communities(adjacency)
        participation = np.full(len(unit_region), np.nan)
        if n_communities:
            for unit in range(len(unit_region)):
                if out_strength[unit] <= 0:
                    continue
                community_strength = np.bincount(
                    community,
                    weights=adjacency[unit],
                    minlength=n_communities,
                )
                probability = community_strength / out_strength[unit]
                participation[unit] = 1.0 - np.square(probability).sum()
        participation_num = np.bincount(
            unit_region,
            weights=np.nan_to_num(participation) * out_strength,
            minlength=N_REGIONS,
        )
        output_participation = np.divide(participation_num, out_region, out=np.full(N_REGIONS, np.nan), where=out_region > 0)

        # Pairwise reciprocal flow, summarized at the root-area level.
        region_adjacency = np.zeros((N_REGIONS, N_REGIONS), dtype=float)
        source_regions, target_regions = np.nonzero(adjacency)
        np.add.at(region_adjacency, (unit_region[source_regions], unit_region[target_regions]), adjacency[source_regions, target_regions])
        reciprocal_weight = np.minimum(region_adjacency, region_adjacency.T).sum(axis=1)
        reciprocity = np.divide(reciprocal_weight, out_region, out=np.full(N_REGIONS, np.nan), where=out_region > 0)

        # Fraction of outgoing unit-edge weight that closes a directed three-node cycle.
        binary = adjacency > 0
        two_step_paths = binary.astype(np.int16) @ binary.astype(np.int16)
        closes_cycle = binary & (two_step_paths.T > 0)
        cycle_numerator = (adjacency * closes_cycle).sum(axis=1)
        cycle_region_num = np.bincount(unit_region, weights=cycle_numerator, minlength=N_REGIONS)
        cycle_participation = np.divide(cycle_region_num, out_region, out=np.full(N_REGIONS, np.nan), where=out_region > 0)

        community_rows.append({
            "Subject": subject,
            "SC_source": sc_source,
            "n_functional_units": len(unit_region),
            "n_global_communities": n_communities,
            "global_unit_modularity_q": global_q,
        })
        for region_id, node in enumerate(REGION_NAMES):
            rows.append({
                "Subject": subject,
                "recording_id": f"subject_{subject}",
                "RegionID": region_id,
                "node": node,
                "FU_DCApost": post_dca[region_id],
                "FU_DCApre": pre_dca[region_id],
                "Hard_OO_fraction": hard_oo[region_id],
                "LogOutIn": log_out_in[region_id],
                "OutputParticipation": output_participation[region_id],
                "Reciprocity": reciprocity[region_id],
                "CycleParticipation": cycle_participation[region_id],
                "inter_out_edges": int(out_region[region_id]),
                "inter_in_edges": int(in_region[region_id]),
                "n_functional_units": int((unit_region == region_id).sum()),
            })

    output_dir = DERIVED_DIR / "functional_unit_region_measures" / sc_source
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "figure12_subject_region_functional_unit_structural_measures_all72.csv"
    current = pd.DataFrame(rows)
    if output.exists():
        current = pd.concat((pd.read_csv(output), current), ignore_index=True)
        current = current.drop_duplicates(subset=("Subject", "RegionID"), keep="last").sort_values(["Subject", "RegionID"])
    current.to_csv(output, index=False)
    community_output = output_dir / "figure12_functional_unit_community_qc.csv"
    communities = pd.DataFrame(community_rows)
    if community_output.exists():
        communities = pd.concat((pd.read_csv(community_output), communities), ignore_index=True)
        communities = communities.drop_duplicates(subset="Subject", keep="last").sort_values("Subject")
    communities.to_csv(community_output, index=False)
    print(f"Saved {output}")
    print(f"Saved {community_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=("cell", "functional_unit"), default="functional_unit")
    parser.add_argument("--sc-source", choices=SC_SOURCE_CHOICES, default="fcs_calibrated_skeleton_kmeans_nearest_r12")
    parser.add_argument("--subjects", nargs="+", type=int, default=CONFIG["subjects"])
    args = parser.parse_args()
    if args.analysis == "cell":
        if args.sc_source != "historical":
            raise ValueError("Cell analysis currently supports only the frozen historical SC")
        main_cell()
    else:
        main_functional_unit(args.sc_source, args.subjects)
