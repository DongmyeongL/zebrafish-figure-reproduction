#!/usr/bin/env python3
"""Generate subject-wise directed SC edge lists from affine-mapped endpoints.

Endpoint coordinates are transformed with the subject-specific full affine
parameters saved by script 26. Endpoint source/target labels can use either
the legacy direct soma-endpoint distance rule or the skeleton-path K-means
rule saved for the revised Figure 12 analysis. Block-overwrite semantics are
preserved in both variants.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial import cKDTree

from skeleton_path_endpoint_classes import save_endpoint_classes


SCRIPT_DIR = Path(__file__).resolve().parent
RECONSTRUCTION_DIR = SCRIPT_DIR / "sc_reconstruction"
PACK_ROOT = SCRIPT_DIR.parents[1]
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK_ROOT / "raw_data"))
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data"))
OUT_DIR = DERIVED_ROOT / "figure12" / "sc_reconstruction"
COMPACT_SC_ROOT = Path(os.environ.get("ZF_COMPACT_SC_ROOT", RAW_ROOT / "figure12"))
COMPACT_SC_DIRS = {
    "fcs_calibrated_endpoint": COMPACT_SC_ROOT / "fcs_calibrated_endpoint_sc",
    "fcs_calibrated_skeleton_path": COMPACT_SC_ROOT / "fcs_calibrated_skeleton_path_sc",
    "fcs_calibrated_skeleton_nearest_endpoint": COMPACT_SC_ROOT / "fcs_calibrated_skeleton_nearest_endpoint_sc",
    "fcs_calibrated_skeleton_kmeans_nearest_r11": COMPACT_SC_ROOT / "fcs_calibrated_skeleton_kmeans_nearest_r11_sc",
    "fcs_calibrated_skeleton_kmeans_nearest_r12": COMPACT_SC_ROOT / "fcs_calibrated_skeleton_kmeans_nearest_r12_sc",
}
PARAMETERS = (
    "m00", "m01", "m02", "bx",
    "m10", "m11", "m12", "by",
    "m20", "m21", "m22", "bz",
)


def load_calibration_module():
    path = RECONSTRUCTION_DIR / "26_calibrate_subject12_transform_to_fcs.py"
    spec = importlib.util.spec_from_file_location("fcs_calibration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def strict_nearby(tree, cell_xyz, transformed, squared_threshold):
    radius = np.sqrt(squared_threshold)
    nearby = tree.query_ball_point(transformed, radius, workers=1)
    result = []
    for endpoint, ids in zip(transformed, nearby):
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size:
            delta = cell_xyz[ids].astype(float) - endpoint
            ids = ids[np.einsum("ij,ij->i", delta, delta) < squared_threshold]
        result.append(ids)
    return result


def nearest_endpoint_assignment(cell_xyz, transformed, squared_threshold):
    """Assign each imaging soma to at most one nearby endpoint in this block."""
    endpoint_tree = cKDTree(np.asarray(transformed, dtype=float))
    distance, endpoint_id = endpoint_tree.query(cell_xyz.astype(float), k=1, workers=1)
    keep = (distance * distance) < squared_threshold
    return [
        np.flatnonzero(keep & (endpoint_id == current_endpoint)).astype(np.int32)
        for current_endpoint in range(len(transformed))
    ]


def uses_skeleton_labels(sc_source: str) -> bool:
    return sc_source in {
        "fcs_calibrated_skeleton_path",
        "fcs_calibrated_skeleton_nearest_endpoint",
        "fcs_calibrated_skeleton_kmeans_nearest_r11",
        "fcs_calibrated_skeleton_kmeans_nearest_r12",
    }


def uses_nearest_endpoint_assignment(sc_source: str) -> bool:
    return sc_source in {
        "fcs_calibrated_skeleton_nearest_endpoint",
        "fcs_calibrated_skeleton_kmeans_nearest_r11",
        "fcs_calibrated_skeleton_kmeans_nearest_r12",
    }


def calibration_tag(sc_source: str) -> str:
    if sc_source in {
        "fcs_calibrated_skeleton_kmeans_nearest_r11",
        "fcs_calibrated_skeleton_kmeans_nearest_r12",
    }:
        return "full_affine_skeleton_path_kmeans"
    return "full_affine"


def generate(subject, squared_threshold, sc_source, skeleton_labels=None):
    calibration = load_calibration_module()
    search = OUT_DIR / f"subject{subject}_fcs_transform_search_{calibration_tag(sc_source)}.csv"
    best = pd.read_csv(search).sort_values("rho_fcs_sc_strength", ascending=False).iloc[0]
    affine = best.loc[list(PARAMETERS)].to_numpy(float).reshape(3, 4)
    cell_xyz, neuron_region = calibration.rebuild_units(subject)
    tree = cKDTree(cell_xyz.astype(float))
    endpoints = sio.loadmat(calibration.ANATOMY_DIR / "neuronEndpoints_data.mat", squeeze_me=True, struct_as_record=False)["neuronEndpoints"]
    soma = None
    if sc_source == "fcs_calibrated_endpoint":
        soma = np.asarray(sio.loadmat(calibration.ANATOMY_DIR / "somaCoordinates_data.mat", squeeze_me=True, struct_as_record=False)["somaCoordinates"])
    output_dir = OUT_DIR / sc_source
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_dir = COMPACT_SC_DIRS[sc_source]

    # One target set per source neuron avoids allocating a dense N x N matrix.
    adjacency = [set() for _ in range(len(cell_xyz))]
    block_rows = []
    for block_id, block in enumerate(endpoints):
        if block_id % 250 == 0:
            print(f"subject {subject}: block {block_id}/{len(endpoints)}", flush=True)
        if np.ndim(block) <= 1:
            continue
        block = np.asarray(block, dtype=float)
        transformed = calibration.transform(block, affine.ravel())
        nearby = (
            nearest_endpoint_assignment(cell_xyz, transformed, squared_threshold)
            if uses_nearest_endpoint_assignment(sc_source)
            else strict_nearby(tree, cell_xyz, transformed, squared_threshold)
        )
        nonempty = [ids for ids in nearby if ids.size]
        if not nonempty:
            continue
        selected = np.unique(np.concatenate(nonempty))
        selected_set = set(selected.tolist())
        # Match ``cellular_sc_mat[np.ix_(selected, selected)] = t_sc_mat``.
        for source in selected:
            adjacency[int(source)].difference_update(selected_set)
        if uses_skeleton_labels(sc_source):
            labels = np.asarray(skeleton_labels[block_id], dtype=bool)
            if len(labels) != len(block):
                raise ValueError(f"Block {block_id}: skeleton labels do not match endpoint count")
        else:
            distance = np.sqrt(
                (soma[block_id, 0] - block[:, 0]) ** 2
                + (soma[block_id, 1] - block[:, 1]) ** 2
                + 4.0 * (soma[block_id, 2] - block[:, 2]) ** 2
            )
            labels = distance < 230.0
        source_chunks = [nearby[i] for i in np.flatnonzero(labels) if nearby[i].size]
        target_chunks = [nearby[i] for i in np.flatnonzero(~labels) if nearby[i].size]
        source_nodes = np.unique(np.concatenate(source_chunks)) if source_chunks else np.empty(0, dtype=np.int32)
        target_nodes = np.unique(np.concatenate(target_chunks)) if target_chunks else np.empty(0, dtype=np.int32)
        source_target_overlap = np.intersect1d(source_nodes, target_nodes, assume_unique=True)
        if source_target_overlap.size:
            raise RuntimeError(f"Block {block_id}: source and target soma assignments overlap")
        if source_chunks and target_chunks:
            target_set = set(target_nodes.tolist())
            for source in source_nodes:
                adjacency[int(source)].update(target_set)
                adjacency[int(source)].discard(int(source))
            n_edges = sum(len(adjacency[int(source)] & target_set) for source in source_nodes)
        else:
            n_edges = 0
        block_rows.append({
            "block_id": block_id,
            "n_endpoints": len(block),
            "n_source_endpoints": int(labels.sum()),
            "n_target_endpoints": int((~labels).sum()),
            "n_selected_cells": len(selected),
            "n_source_cells": len(source_nodes),
            "n_target_cells": len(target_nodes),
            "n_source_target_overlap": int(source_target_overlap.size),
            "n_block_edges": n_edges,
        })

    counts = np.fromiter((len(targets) for targets in adjacency), dtype=np.int64)
    source = np.repeat(np.arange(len(adjacency), dtype=np.int32), counts)
    target = np.fromiter((target for targets in adjacency for target in sorted(targets)), dtype=np.int32, count=int(counts.sum()))
    output = output_dir / f"subject_{subject}_full_affine_endpoint_edges.npz"
    np.savez_compressed(
        output,
        source=source,
        target=target,
        CellXYZ=cell_xyz.astype(np.int32),
        neuron_root_area=neuron_region.astype(np.int16),
        affine_matrix=affine,
        squared_distance_threshold=np.asarray(squared_threshold),
        soma_distance_threshold=np.asarray(230.0),
        endpoint_label_rule=np.asarray(
            "skeleton_path_kmeans_nearest_endpoint_assignment"
            if uses_nearest_endpoint_assignment(sc_source)
            else "skeleton_path_kmeans"
            if sc_source == "fcs_calibrated_skeleton_path"
            else "direct_soma_distance"
        ),
        interpolation_enabled=np.asarray(False),
    )
    # Match the canonical Figure 12 compact-SC schema without replacing the
    # frozen historical inputs in ``raw_data/figure12``.
    compact_dir.mkdir(parents=True, exist_ok=True)
    compact = compact_dir / f"subject_{subject}_compact_sc.npz"
    np.savez_compressed(
        compact,
        edges=np.column_stack((source, target)),
        neuron_region=neuron_region.astype(np.int16),
        source_file=np.asarray(str(output)),
    )
    qc = output_dir / f"subject_{subject}_full_affine_endpoint_edges_qc.csv"
    pd.DataFrame(block_rows).to_csv(qc, index=False)
    return output, compact, qc, len(source), len(cell_xyz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(12, 19)))
    parser.add_argument(
        "--sc-source",
        choices=tuple(COMPACT_SC_DIRS),
        default="fcs_calibrated_skeleton_kmeans_nearest_r12",
    )
    parser.add_argument("--squared-threshold", type=float, default=144.0)
    args = parser.parse_args()
    skeleton_labels = None
    if uses_skeleton_labels(args.sc_source):
        _, skeleton_labels, _ = save_endpoint_classes(OUT_DIR / args.sc_source)
    rows = []
    for subject in args.subjects:
        output, compact, qc, n_edges, n_cells = generate(subject, args.squared_threshold, args.sc_source, skeleton_labels)
        rows.append({"subject": subject, "sc_source": args.sc_source, "n_cells": n_cells, "n_endpoint_edges": n_edges, "edge_file": str(output), "compact_sc_file": str(compact), "qc_file": str(qc)})
        print(f"subject {subject}: edges={n_edges:,}", flush=True)
    summary = OUT_DIR / args.sc_source / "full_affine_endpoint_edge_generation_summary.csv"
    current = pd.DataFrame(rows)
    if summary.exists():
        current = pd.concat((pd.read_csv(summary), current), ignore_index=True)
        current = current.drop_duplicates(subset="subject", keep="last").sort_values("subject")
    current.to_csv(summary, index=False)
    current.to_csv(COMPACT_SC_DIRS[args.sc_source] / f"{args.sc_source}_manifest.csv", index=False)
    print(f"saved={summary}")


if __name__ == "__main__":
    main()
