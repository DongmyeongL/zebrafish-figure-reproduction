#!/usr/bin/env python3
"""Extract the minimal immutable SC arrays needed by Figure 12."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from common import CONFIG, RAW_DIR, compact_sc_path, ensure_output_dirs, original_sc_path


def main() -> None:
    ensure_output_dirs()
    qc_rows = []
    for subject in CONFIG["subjects"]:
        source = original_sc_path(subject)
        if not source.exists():
            raise FileNotFoundError(source)
        print(f"[Figure 12] subject {subject}: extracting compact SC", flush=True)
        with source.open("rb") as handle:
            raw = pickle.load(handle)

        edges = np.asarray(raw["cellular_sc_list"], dtype=np.int64)
        root_area = np.asarray(raw["root_area"], dtype=np.int64)
        neuron_region_id = np.asarray(raw["neuron_region_id"], dtype=np.int64)
        neuron_region = root_area[neuron_region_id]
        n_neurons = int(len(neuron_region))
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"Subject {subject}: expected an edge-by-2 cellular_sc_list")
        if len(edges) and (edges.min() < 0 or edges.max() >= n_neurons):
            raise ValueError(f"Subject {subject}: edge endpoint outside neuron range")

        np.savez(
            compact_sc_path(subject),
            edges=edges,
            neuron_region=neuron_region,
            source_file=np.asarray(str(source)),
        )
        qc_rows.append(
            {
                "Subject": subject,
                "n_neurons": n_neurons,
                "n_edges": int(len(edges)),
                "n_regions_present": int(np.unique(neuron_region).size),
                "source_file": str(source),
            }
        )
        del raw, edges, root_area, neuron_region_id, neuron_region

    output = RAW_DIR / "figure12_compact_sc_qc.csv"
    pd.DataFrame(qc_rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
