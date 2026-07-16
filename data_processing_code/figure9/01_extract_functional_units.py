#!/usr/bin/env python3
"""Stage the saved spatial functional units used by Figure 9.

The default mode copies the frozen, compact final_id_cluster trace files into
derived_data. Passing --from-original regenerates those files from the large
cell-level subject PKLs and the saved final_id_cluster assignments.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import CONFIG, DERIVED_DIR, RAW_DIR, ensure_output_dirs


def extract_from_original(source: Path, labels: pd.DataFrame, destination: Path) -> dict:
    with source.open("rb") as handle:
        raw = pickle.load(handle)

    id_to_label = dict(zip(labels["RegionID"].astype(int), labels["node"].astype(str)))
    spot = np.asarray(raw["spot_data"], dtype=np.float32)
    traces: list[np.ndarray] = []
    cluster_ids: list[int] = []
    root_area_ids: list[int] = []
    root_area_names: list[str] = []
    n_cells: list[int] = []

    for cluster_id, (region_id, members) in enumerate(
        zip(raw["root_area"], raw["final_id_cluster"], strict=False)
    ):
        region_id = int(region_id)
        if region_id not in id_to_label:
            continue
        member_ids = [int(idx) for idx in members if 0 <= int(idx) < spot.shape[0]]
        if not member_ids:
            continue
        traces.append(np.nanmean(spot[member_ids], axis=0))
        cluster_ids.append(cluster_id)
        root_area_ids.append(region_id)
        root_area_names.append(id_to_label[region_id])
        n_cells.append(len(member_ids))

    if not traces:
        raise RuntimeError(f"No functional units extracted from {source}")

    # Match the canonical export order: atlas/root-area order first, then the
    # saved final_id_cluster identifier within each root area.
    region_order = {int(region_id): pos for pos, region_id in enumerate(labels["RegionID"].astype(int))}
    order = sorted(
        range(len(cluster_ids)),
        key=lambda idx: (region_order[root_area_ids[idx]], cluster_ids[idx]),
    )
    traces = [traces[idx] for idx in order]
    cluster_ids = [cluster_ids[idx] for idx in order]
    root_area_ids = [root_area_ids[idx] for idx in order]
    root_area_names = [root_area_names[idx] for idx in order]
    n_cells = [n_cells[idx] for idx in order]

    subject = int(source.name.split("_")[1])
    payload = {
        "recording_id": f"subject_{subject}",
        "subject": subject,
        "traces": np.vstack(traces).astype(np.float32, copy=False),
        "trace_axis": "final_id_clusters x time",
        "cluster_ids": cluster_ids,
        "root_area_ids": root_area_ids,
        "root_area_names": root_area_names,
        "n_cells_per_cluster": n_cells,
        "sampling_rate_hz": float(CONFIG["sampling_rate_hz"]),
        "source_raw_path": str(source),
    }
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-original",
        action="store_true",
        help="Regenerate final_id_cluster traces from the large original subject PKLs.",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    out_dir = DERIVED_DIR / "functional_unit_traces"
    manifest_rows = []
    labels = pd.read_csv(RAW_DIR / "region_labels.csv")

    for subject in CONFIG["subjects"]:
        destination = out_dir / f"zebrafish_subject_{subject}_raw_cluster_traces.pkl"
        if args.from_original:
            source = RAW_DIR / "original_subjects" / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
            payload = extract_from_original(source, labels, destination)
            source_mode = "regenerated_from_original_subject_pkl"
        else:
            source = RAW_DIR / "functional_unit_traces" / destination.name
            shutil.copy2(source, destination)
            with destination.open("rb") as handle:
                payload = pickle.load(handle)
            source_mode = "frozen_final_id_cluster_trace"

        traces = np.asarray(payload["traces"])
        manifest_rows.append(
            {
                "recording_id": payload["recording_id"],
                "source_mode": source_mode,
                "source": str(source.relative_to(RAW_DIR)),
                "output": str(destination.relative_to(DERIVED_DIR)),
                "n_functional_units": int(traces.shape[0]),
                "n_timepoints": int(traces.shape[1]),
                "n_root_areas": int(len(set(payload["root_area_ids"]))),
                "sampling_rate_hz": float(payload["sampling_rate_hz"]),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DERIVED_DIR / "figure9_functional_unit_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
