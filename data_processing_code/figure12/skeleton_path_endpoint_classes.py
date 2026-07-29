"""Per-morphology endpoint classes from skeleton-path length clustering."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans


PACK_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK_ROOT / "raw_data"))
RAW_DIR = Path(os.environ.get("ZF_ANATOMY_ROOT", RAW_ROOT / "figure12" / "anatomy"))
RANDOM_SEED = 42


def path_lengths(vertices, segments, morphology_soma, endpoints):
    """Shortest soma-to-endpoint distance along the single-neuron skeleton."""
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    endpoints = np.asarray(endpoints, dtype=float)
    segments = np.asarray(segments, dtype=np.int64) - 1  # MATLAB indexing
    valid = (
        (segments[:, 0] >= 0) & (segments[:, 0] < len(vertices))
        & (segments[:, 1] >= 0) & (segments[:, 1] < len(vertices))
    )
    segments = segments[valid]
    if len(vertices) == 0 or len(segments) == 0:
        return np.full(len(endpoints), np.inf)

    delta = vertices[segments[:, 0]] - vertices[segments[:, 1]]
    delta[:, 2] *= 2.0
    edge_length = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    graph = coo_matrix(
        (
            np.concatenate((edge_length, edge_length)),
            (
                np.concatenate((segments[:, 0], segments[:, 1])),
                np.concatenate((segments[:, 1], segments[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    soma_delta = vertices - np.asarray(morphology_soma, dtype=float)[:3]
    soma_delta[:, 2] *= 2.0
    soma_vertex = int(np.argmin(np.einsum("ij,ij->i", soma_delta, soma_delta)))
    endpoint_delta = vertices[None, :, :] - endpoints[:, None, :3]
    endpoint_delta[:, :, 2] *= 2.0
    endpoint_vertices = np.argmin(np.sum(endpoint_delta * endpoint_delta, axis=2), axis=1)
    return dijkstra(graph, directed=False, indices=soma_vertex)[endpoint_vertices]


def always_split(path):
    """Always produce proximal/source and distal/target labels when n >= 2."""
    source = np.zeros(len(path), dtype=bool)
    finite = np.flatnonzero(np.isfinite(path))
    if len(finite) <= 1:
        source[finite] = True
        return source
    values = path[finite, None]
    labels = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=25).fit_predict(values)
    centers = np.array([values[labels == cluster].mean() for cluster in range(2)])
    if np.isclose(centers[0], centers[1]):
        ordered = finite[np.argsort(path[finite])]
        source[ordered[: max(1, len(ordered) // 2)]] = True
    else:
        source[finite[labels == int(np.argmin(centers))]] = True
    # Skeleton-disconnected endpoints remain distal/target.
    return source


def build_endpoint_classes(raw_dir: Path = RAW_DIR):
    """Return raw endpoint blocks, per-endpoint source labels, and path lengths."""
    raw_dir = Path(raw_dir)
    morphology = sio.loadmat(raw_dir / "signle_neuron_poistion_data.mat", squeeze_me=True, struct_as_record=False)
    soma = np.asarray(sio.loadmat(raw_dir / "somaCoordinates_data.mat", squeeze_me=True, struct_as_record=False)["somaCoordinates"])
    endpoint_blocks = sio.loadmat(raw_dir / "neuronEndpoints_data.mat", squeeze_me=True, struct_as_record=False)["neuronEndpoints"]
    data_xyz = morphology["data_xyz"]
    connection_line = morphology["connection_line"]
    labels, paths = [], []
    for block_id, block in enumerate(endpoint_blocks):
        if np.ndim(block) <= 1:
            labels.append(np.empty(0, dtype=bool))
            paths.append(np.empty(0, dtype=float))
            continue
        block = np.asarray(block, dtype=float)
        path = path_lengths(data_xyz[block_id], connection_line[block_id], soma[block_id], block)
        labels.append(always_split(path))
        paths.append(path)
    return endpoint_blocks, labels, paths


def save_endpoint_classes(output_dir: Path):
    """Compute once and save endpoint-level and block-level provenance tables."""
    endpoint_blocks, labels, paths = build_endpoint_classes()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for block_id, (block, source, path) in enumerate(zip(endpoint_blocks, labels, paths)):
        if np.ndim(block) <= 1:
            continue
        finite = np.isfinite(path)
        source_center = np.nanmean(path[source & finite]) if np.any(source & finite) else np.nan
        target_center = np.nanmean(path[(~source) & finite]) if np.any((~source) & finite) else np.nan
        summary.append({
            "block_id": block_id,
            "n_endpoints": len(path),
            "n_source_endpoints": int(source.sum()),
            "n_target_endpoints": int((~source).sum()),
            "n_reachable_endpoints": int(finite.sum()),
            "source_path_center": source_center,
            "target_path_center": target_center,
        })
        rows.extend({
            "block_id": block_id,
            "endpoint_id": endpoint_id,
            "skeleton_path_length": float(value),
            "endpoint_class": "source" if source[endpoint_id] else "target",
            "path_reachable": bool(np.isfinite(value)),
        } for endpoint_id, value in enumerate(path))
    endpoint_table = pd.DataFrame(rows)
    endpoint_table.to_csv(output_dir / "skeleton_path_endpoint_cluster_assignments.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "skeleton_path_endpoint_cluster_summary.csv", index=False)
    return endpoint_blocks, labels, paths
