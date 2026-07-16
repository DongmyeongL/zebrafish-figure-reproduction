# Raw-to-derived input contract

The public workflow starts from **prepared raw inputs**, not directly from
microscope files. Prepared raw inputs retain cell-, functional-unit-, ROI-, or
edge-level observations but have already undergone dataset-specific export,
atlas registration, or annotation matching. Paths are relative to the
directory passed through `--raw-root`.

## Bundled inputs

### Zebrafish functional-unit traces

`raw_data/figure9/functional_unit_traces/` contains seven pickle files. Each
contains `traces`, `cluster_ids`, `root_area_ids`, `root_area_names`,
`n_cells_per_cluster`, `recording_id`, and `sampling_rate_hz`. These files are
sufficient to recompute FCV, FCS, FC reconfiguration degree, NetTE, and
neighborhood NetTE for Figure 9 and Supply 13.

### Compact and frozen supplementary inputs

- Supply 1 contains histogram counts and distance-bin lognormal parameters.
- Supply 5 contains fixed network diagrams and null-statistic arrays.
- Figure 13/Supply 15 contain frozen whole-brain perturbation summaries and
  representative traces. The four-layer model itself can be rerun from code.

## External zebrafish inputs

Seven prepared subject bundles are expected at:

```text
figure9/original_subjects/subject_12_data_cellular_synapse_sc_100_data.pkl
...
figure9/original_subjects/subject_18_data_cellular_synapse_sc_100_data.pkl
```

The raw-to-derived scripts use these keys:

- `spot_data`, `final_id_cluster`, and `root_area` for functional units;
- `cellular_sc_list`, `neuron_region_id`, and `root_area` for directed SC;
- `stim_data`, `stim_array`, `neuron_region_id`, and `root_area` for stimulus
  FCV/FCS.

Alternatively, Figure 12 can begin from seven compact files at
`figure12/subject_<ID>_compact_sc.npz`, each containing `edges` and
`neuron_region`. The compact files total approximately 1.1 GB and are therefore
not stored in GitHub.

## External C. elegans inputs

Expected under `invertebrates/celegans/`:

- `recordings/*_raw_traces.pkl`, containing `traces_spontaneous`,
  `neuron_names`, `sampling_rate_hz`, and `recording_id`;
- `herm_full_edgelist.csv`, the directed chemical-synapse edge list;
- the bundled cell-class and fine-class annotation CSV files.

## External Drosophila inputs

Expected under `invertebrates/drosophila/`:

- `recordings/*_raw_traces.pkl`, containing ROI traces, side-aware atlas labels,
  sampling rate, and recording identifier;
- `proofread_connections_783.feather`;
- `proofread_root_ids_783.npy`;
- pre- and postsynaptic neuropil-localization feather tables;
- `ito_region_order.csv`.

These inputs are prepared exports from FlyWire783 and Branson999. Consult
`config/datasets.csv` for source publications and required paths.

## Data placement audit

```bash
python scripts/manage_data.py
python scripts/manage_data.py --target figure12
python scripts/manage_data.py --target celegans
```

Direct download URLs are intentionally left blank where the source provider
does not expose the prepared export as a stable file. Before redistribution,
confirm the license of each primary dataset and deposit the prepared large
files in a DOI-backed archive such as Zenodo or OSF.
