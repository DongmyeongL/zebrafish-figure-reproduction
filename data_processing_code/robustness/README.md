# SI robustness analyses

These scripts reproduce the numerical summaries used in the SI robustness
sections for the primary zebrafish structural-connectivity reconstruction:
skeleton-path KMeans endpoint classification, nearest-endpoint soma assignment,
an endpoint-matching radius of 12 imaging-coordinate units, and a target size of
400 neurons per anatomical unit.

Two reproduction levels are provided. The default command below starts from
bundled iteration- and region-level tables. The full-generation command starts
from the external soma, endpoint, morphology, and prepared subject files and
regenerates the reconstruction-dependent iteration tables.

Run all analyses from the repository root with:

```bash
python data_processing_code/robustness/run_all.py
```

The scripts write tables to `statistics/robustness/`.

## Full generation from source data

Arrange the external data as specified in `RAW_DATA.md`, then run:

```bash
python data_processing_code/robustness/generate_full_reconstruction_controls.py \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/robustness_full \
  --output work/robustness_inputs
```

This command performs the following operations with the manuscript defaults:

1. calibrates subject-specific full-affine transforms and reconstructs the
   skeleton-r12 endpoint-derived SC;
2. rebuilds SC and regional DCA/OO values for all 30 anatomical-unit-size and
   endpoint-radius combinations;
3. assigns directed anatomical-unit relations to their supporting reconstructed
   morphologies and draws 200 subsets at each incomplete retained fraction;
4. generates 250 weighted maximum-entropy nulls that preserve every anatomical
   unit's incoming and outgoing strength; and
5. exports the resulting iteration tables in the compact public schema.

The calculation is CPU- and memory-intensive. Existing calibration and
endpoint-edge files can be reused with `--skip-sc-reconstruction`. Reduced
`--subjects`, `--n-null`, and `--subsample-iterations` values are intended only
for smoke testing, not manuscript reproduction.

To summarize newly generated tables, point the analysis scripts to them:

```bash
ZF_ROBUSTNESS_DATA_ROOT=work/robustness_inputs \
ZF_ROBUSTNESS_STATS_ROOT=work/robustness_statistics \
python data_processing_code/robustness/01_reconstruction_parameter_sensitivity.py
```

| Script | SI control | Released input |
|---|---|---|
| `01_reconstruction_parameter_sensitivity.py` | anatomical-unit size and endpoint radius | correlations for the 30 reconstruction settings |
| `02_morphology_subsampling.py` | random subsampling of reconstructed morphologies | 200 iteration-level results per incomplete fraction |
| `03_strength_preserving_topology_null.py` | weighted in/out-strength-preserving topology null | 250 null correlations and observed correlations |
| `04_division_and_subject_controls.py` | division adjustment and subject-specific meta-analysis | region and subject-by-region analysis tables |
| `05_prediction_residual_controls.py` | measurement and sampling proxies | 41-region predictor and proxy table, excluding OB |

## Bundled reconstruction-dependent inputs

The parameter-grid, morphology-subsampling, and topology-null inputs were
generated from the full soma and reconstructed-morphology endpoint data. Those
raw coordinate arrays are not distributed in this lightweight repository.
Their released iteration-level tables permit a fast numerical check, while
`generate_full_reconstruction_controls.py` and the generators in
`data_processing_code/figure12/validation/` expose the complete calculation
that created them.

The strength-preserving null retained each anatomical unit's weighted incoming
and outgoing strength, prohibited within-region edges, and randomized weight
allocation over admissible inter-regional pairs. The released file also
contains exploratory root-pair-preserving realizations; the public analysis
script explicitly selects only `null_type == "strength_preserving"`, matching
the SI analysis.
