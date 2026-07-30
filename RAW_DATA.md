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

- Supply 1 contains the final six-panel QC asset and its fit-statistics table;
  the underlying microscopy-scale morphology and endpoint MAT files remain external.
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

Figure 12 can begin from seven r12 compact files at
`figure12/fcs_calibrated_skeleton_kmeans_nearest_r12_sc/subject_<ID>_compact_sc.npz`,
each containing `edges` and `neuron_region`. To reconstruct those files, place
subject coordinate MAT files under `figure12/original_subject_mat/` and
`neuronEndpoints_data.mat`, `somaCoordinates_data.mat`, and
`signle_neuron_poistion_data.mat` under `figure12/anatomy/`. The public pipeline
then performs subject-specific FCS calibration, skeleton-path K-means endpoint
classification, nearest-endpoint assignment within a radius of 12 imaging-coordinate
units, anatomical-unit DCA, and regional aggregation. These large external inputs
and generated compact SC files are not stored in GitHub.

### SI robustness reconstruction

The complete reconstruction-dependent SI controls use the same three external
input groups: prepared subject bundles, subject coordinate MAT files, and the
three anatomy/morphology MAT files. The full driver recalibrates the affine
alignment, reconstructs skeleton-r12 endpoint edges, evaluates all 30
anatomical-unit-size/radius settings, samples reconstructed morphologies, and
generates weighted strength-preserving null networks:

```bash
python data_processing_code/robustness/generate_full_reconstruction_controls.py \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/robustness_full \
  --output work/robustness_inputs
```

Audit these inputs before a full run with:

```bash
python scripts/manage_data.py --raw-root /path/to/prepared_raw_data \
  --target robustness --strict
```

The OO-threshold analysis additionally uses the subject r12 compact SC,
anatomical-unit DCA files, and prepared subject bundles that define
anatomical-unit membership. Its full generator is:

```bash
ZF_RAW_DATA_ROOT=/path/to/prepared_raw_data \
ZF_ORIGINAL_SC_DIR=/path/to/prepared_raw_data/figure9/original_subjects \
ZF_DERIVED_DATA_ROOT=work/robustness_full/derived_data \
ZF_OO_SENSITIVITY_DERIVED=work/robustness_inputs/oo_threshold_sensitivity \
ZF_OO_SENSITIVITY_STATS=work/robustness_statistics \
ZF_OO_SENSITIVITY_FIGURES=work/robustness_figures \
python data_processing_code/figure12/validation/20_r12_oo_threshold_sensitivity.py
```

Git includes the resulting compact subject-region threshold, near-zero
exclusion, and soft-OO tables, so the reported statistical summaries can be
reproduced without redistributing the cell-level SC. The OMR adjustment uses
the bundled matched spontaneous/stimulus region table and requires no
additional external input. Its standalone analysis is also available as
`data_processing_code/figure12/validation/21_r12_omr_spontaneous_adjustment.py`;
`ZF_OMR_ADJUSTMENT_INPUT` and `ZF_OMR_ADJUSTMENT_OUTPUT` can be used to select
alternative input and output paths.

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
