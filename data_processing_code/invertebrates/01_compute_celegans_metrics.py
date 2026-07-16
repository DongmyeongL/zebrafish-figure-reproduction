"""Compute C. elegans FCV, DCA, Pre/Post-DCA, and OO fraction from raw data."""

from __future__ import annotations

import pickle
import re
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PACK = HERE.parents[2]
RAW = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK / "raw_data")).resolve() / "invertebrates" / "celegans"
OUT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data")).resolve() / "invertebrates"
sys.path.insert(0, str(HERE.parent))

from common_metrics import edge_std_fcv, preprocess_traces, rank1_dca_dense, zscore_finite

WINDOW = 20
STEP = 8
HIGHPASS_HZ = 0.03


def structural_metrics() -> pd.DataFrame:
    edges = pd.read_csv(RAW / "herm_full_edgelist.csv")
    edges["Source"] = edges["Source"].astype(str).str.strip()
    edges["Target"] = edges["Target"].astype(str).str.strip()
    normalize = lambda name: re.sub(r"^([A-Z]+)0+(\d+)$", r"\1\2", name)
    edges["Source"] = edges["Source"].map(normalize)
    edges["Target"] = edges["Target"].map(normalize)
    weight_col = next((c for c in ["Weight", "weight", "Nbr", "Number"] if c in edges), None)
    edges["_weight"] = pd.to_numeric(edges[weight_col], errors="coerce").fillna(1.0) if weight_col else 1.0
    classes = pd.read_csv(RAW / "celegans_cell_classes.csv")
    valid_neurons = set(classes.loc[
        classes["cell_class"].isin({"Sensory", "Interneuron", "Motorneuron"}), "node"
    ].astype(str))
    edges = edges[
        edges["Type"].astype(str).str.strip().eq("chemical")
        & edges["Source"].isin(valid_neurons)
        & edges["Target"].isin(valid_neurons)
        & edges["_weight"].gt(0)
    ].copy()
    neurons = sorted(set(edges["Source"]) | set(edges["Target"]))
    index = {name: i for i, name in enumerate(neurons)}
    sc = np.zeros((len(neurons), len(neurons)), dtype=float)
    for source, target, weight in edges[["Source", "Target", "_weight"]].itertuples(index=False, name=None):
        sc[index[source], index[target]] += weight
    np.fill_diagonal(sc, 0.0)
    c_out, c_in, dca = rank1_dca_dense(sc)
    binary = sc > 0
    out_strength = sc.sum(axis=1)
    in_strength = sc.sum(axis=0)
    post = np.divide(sc @ dca, out_strength, out=np.full(len(sc), np.nan), where=out_strength > 0)
    pre = np.divide(sc.T @ dca, in_strength, out=np.full(len(sc), np.nan), where=in_strength > 0)
    output_like = dca > 0
    oo = np.full(len(sc), np.nan)
    for i in range(len(sc)):
        targets = np.flatnonzero(binary[i])
        if targets.size:
            oo[i] = np.mean(output_like[i] & output_like[targets])
    return pd.DataFrame({"node": neurons, "DCA": dca, "PostDCA": post, "PreDCA": pre, "OO_fraction": oo,
                         "out_degree": binary.sum(1), "in_degree": binary.sum(0)})


def functional_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((RAW / "recordings").glob("*_raw_traces.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        traces = preprocess_traces(raw["traces_spontaneous"], float(raw["sampling_rate_hz"]), HIGHPASS_HZ)
        fcv = edge_std_fcv(traces, WINDOW, STEP)
        fcv_z = zscore_finite(fcv)
        rows.extend({"recording_id": raw["recording_id"], "node": node, "FCV_raw": value,
                     "EdgeStdFCV": zvalue, "n_nodes": len(fcv)}
                    for node, value, zvalue in zip(raw["neuron_names"], fcv, fcv_z))
    recording = pd.DataFrame(rows)
    summary = recording.groupby("node", as_index=False).agg(
        EdgeStdFCV=("EdgeStdFCV", "mean"), FCV_raw=("FCV_raw", "mean"), n_recordings=("recording_id", "nunique"))
    return recording, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recording, functional = functional_metrics()
    structural = structural_metrics()
    annotations = pd.read_csv(RAW / "celegans_fine_class_annotations.csv")[["node", "fine_class"]].drop_duplicates("node")
    annotations["fine_class"] = annotations["fine_class"].replace({
        "mechanosensory": "mechanosensory / other sensory",
        "other sensory neuron": "mechanosensory / other sensory",
        "head motor / command": "head motor / premotor",
        "other motor neuron": "head motor / premotor",
        "integrative interneuron": "interneuron / integrative",
        "other interneuron": "interneuron / integrative",
    })
    merged = functional.merge(structural, on="node", how="inner").merge(annotations, on="node", how="left")
    recording.to_csv(OUT / "celegans_fcv_recording_node.csv", index=False)
    structural.to_csv(OUT / "celegans_structural_node_metrics.csv", index=False)
    merged.to_csv(OUT / "celegans_node_metrics.csv", index=False)
    print(f"C. elegans: {len(recording)} recording-node rows, {len(merged)} matched neurons")


if __name__ == "__main__":
    main()
