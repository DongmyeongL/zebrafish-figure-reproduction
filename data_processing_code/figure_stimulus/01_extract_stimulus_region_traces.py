#!/usr/bin/env python3
"""Extract compact root-area mean traces from the raw OMR recordings."""

from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd

from common import (
    ANALYSIS_REGIONS_FILE,
    CONFIG,
    DERIVED_DIR,
    LABELS_FILE,
    SOURCE_DIR,
    ensure_output_dirs,
)


def extract_subject(subject: int, labels: pd.DataFrame) -> dict:
    source = SOURCE_DIR / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
    with source.open("rb") as handle:
        raw = pickle.load(handle)

    cell_traces = np.asarray(raw["stim_data"], dtype=np.float32)
    neuron_region_id = np.asarray(raw["neuron_region_id"], dtype=int)
    root_area = np.asarray(raw["root_area"], dtype=int)
    neuron_root_area = root_area[neuron_region_id]
    region_ids = labels["RegionID"].to_numpy(dtype=int)

    region_traces = np.full(
        (len(region_ids), cell_traces.shape[1]), np.nan, dtype=np.float32
    )
    neuron_counts = np.zeros(len(region_ids), dtype=int)
    for position, region_id in enumerate(region_ids):
        members = np.flatnonzero(neuron_root_area == region_id)
        neuron_counts[position] = members.size
        if members.size:
            region_traces[position] = np.nanmean(cell_traces[members], axis=0)

    stimulus = np.asarray(raw["stim_array"]).ravel().astype(np.int16, copy=False)
    if stimulus.size != region_traces.shape[1]:
        raise RuntimeError(
            f"Subject {subject}: stimulus length {stimulus.size} does not match "
            f"trace length {region_traces.shape[1]}"
        )

    destination = (
        DERIVED_DIR / "region_traces" / f"subject_{subject}_stimulus_region_traces.npz"
    )
    np.savez_compressed(
        destination,
        subject=np.asarray(subject),
        region_ids=region_ids,
        region_names=np.asarray(labels["node"].astype(str).tolist(), dtype="U16"),
        region_traces=region_traces,
        neuron_counts=neuron_counts,
        stimulus_array=stimulus,
    )
    result = {
        "subject": subject,
        "source": str(source),
        "output": str(destination),
        "n_regions": len(region_ids),
        "n_regions_with_neurons": int(np.count_nonzero(neuron_counts)),
        "n_neurons": int(neuron_counts.sum()),
        "n_timepoints": int(stimulus.size),
    }
    for stimulus_index in CONFIG["stimulus_indices"]:
        result[f"n_frames_stimulus_{stimulus_index}"] = int(
            np.count_nonzero(stimulus == stimulus_index)
        )
    del raw, cell_traces, region_traces
    gc.collect()
    return result


def main() -> None:
    ensure_output_dirs()
    atlas_labels = pd.read_csv(LABELS_FILE)
    analysis_regions = pd.read_csv(ANALYSIS_REGIONS_FILE)[["root_area_id", "node"]]
    labels = (
        atlas_labels.merge(
            analysis_regions,
            left_on=["RegionID", "node"],
            right_on=["root_area_id", "node"],
            how="inner",
            validate="one_to_one",
        )[["RegionID", "node"]]
        .sort_values("RegionID")
        .reset_index(drop=True)
    )
    if len(labels) != 66:
        raise RuntimeError(f"Expected the 66 Figure 9 analysis regions, found {len(labels)}")
    if labels["RegionID"].duplicated().any() or labels["node"].duplicated().any():
        raise RuntimeError("Region label table must be one-to-one")
    rows = []
    for subject in CONFIG["subjects"]:
        rows.append(extract_subject(int(subject), labels))
        print(f"subject {subject}: extracted root-area stimulus traces", flush=True)
    pd.DataFrame(rows).to_csv(DERIVED_DIR / "stimulus_trace_manifest.csv", index=False)


if __name__ == "__main__":
    main()
