# Manuscript figure reproduction bundle

This directory contains the plotting code and compact, frozen analysis inputs
needed to reproduce six main-text figures and eleven supplementary figures. It
also contains public processing scripts that rebuild derived tables from
prepared raw inputs. Large calcium-imaging and cell-level connectome exports
are not redistributed in Git; bundled functional-unit traces provide one
self-contained raw-to-derived example, while the remaining inputs can be
supplied through an external data directory.

The conceptual schematic `fig1.png` is intentionally outside this bundle; it
is an illustration rather than a data-derived figure.

## Rebuild all figures

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all_figures.py
python validate_release.py
```

PNG outputs are written to `figures/` at manuscript resolution. Statistical tables generated
by the plotting scripts are written to `statistics/`.
Generated figures, statistics, and intermediate plotting tables are ignored by
Git; run the commands above to recreate them locally. `checksums.sha256`
records the integrity of the distributed code and frozen inputs.
Maintainers can regenerate it after intentional release changes with
`python scripts/update_checksums.py`.

### Tested environment

The release was tested on 64-bit Linux with Python 3.10. A GPU is not
required. Rebuilding all figures takes approximately one minute on the system
used to prepare this release, although runtime depends on hardware and font
availability. Matplotlib may fall back to DejaVu Sans when Helvetica or Nimbus
Sans is unavailable; this does not affect the numerical results.

### Rebuild one figure

Each plotting script can also be run independently from the repository root.
For example:

```bash
python figure_code/figure9_final_v2.py
python figure_code/figure12_final_v3.py
python figure_code/figure13_final_v2.py
python figure_code/figure_supply_15.py
```

To verify the distributed files before running the analysis, use:

```bash
sha256sum -c checksums.sha256
```

## Rebuild derived data

The public processing workflow starts from prepared raw inputs rather than
microscope-native files. Audit the available inputs with:

```bash
python scripts/manage_data.py
```

Figure 9 can be rebuilt from the seven compact functional-unit trace files
distributed with this repository:

```bash
python scripts/run_pipeline.py \
  --target figure9 \
  --stage derived \
  --derived-root work/figure9/derived_data
```

For analyses requiring externally stored data, place the inputs in the layout
documented in `RAW_DATA.md` and run, for example:

```bash
python scripts/run_pipeline.py \
  --target figure12 \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/figure12/derived_data

python scripts/run_pipeline.py \
  --target stimulus \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/stimulus/derived_data
```

Available targets are `figure9`, `figure12`, `stimulus`, `celegans`,
`drosophila`, `layer`, and `supply13`. The complete layer-model and
TE-surrogate calculations require `--full-model` and `--full-controls`,
respectively, because they are computationally intensive. Use `--skip-missing`
with `--target all` to run only targets whose inputs are available.

Derived outputs can be checked independently of private reference files:

```bash
python scripts/validate_derived.py figure9 \
  --derived-root work/figure9/derived_data \
  --compare-bundled
python tests/smoke_test.py
```

### Reproduce SI robustness statistics

Compact iteration- and region-level inputs are included for the SI controls of
the primary skeleton-r12 structural analysis. Run all eight robustness summaries
with:

```bash
python data_processing_code/robustness/run_all.py
```

This reproduces the anatomical-unit-size/radius grid, reconstructed-morphology
subsampling, strength-preserving topology null, division-adjusted analysis,
subject-specific random-effects meta-analysis, prediction-residual controls,
OO-threshold/soft-OO sensitivity, and spontaneous-FCV-adjusted OMR analyses.
Outputs are written to `statistics/robustness/`. See
`data_processing_code/robustness/README.md` for the exact sampling units,
released inputs, and the boundary between full reconstruction and compact
statistical reproduction.

The reconstruction-dependent robustness inputs can also be regenerated from
the external Figure 12 soma, endpoint, morphology, and prepared subject data:

```bash
python data_processing_code/robustness/generate_full_reconstruction_controls.py \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/robustness_full \
  --output work/robustness_inputs
```

The input contract, expected schemas, source links, and redistribution
boundaries are documented in `RAW_DATA.md`, `DATA_SOURCES.md`, and
`config/datasets.csv`.

## Main-figure map

The internal filenames retain their development names. The manuscript mapping
is as follows.

| Manuscript figure | Output | Plotting code | Primary bundled input | Reproduction level |
|---|---|---|---|---|
| Fig. 2 | `figure9_final_v2.png` | `figure_code/figure9_final_v2.py` | `derived_data/figure9/` | Bundled prepared raw to derived table to figure |
| Fig. 3 | `figure12_final_v3_fcs_calibrated_skeleton_kmeans_nearest_r12.png` | `figure_code/figure12_final_v3.py` | `derived_data/figure12/functional_unit_region_measures/fcs_calibrated_skeleton_kmeans_nearest_r12/` | Subject-region structural table to figure |
| Fig. 4 | `figure_fcv_fcs_fu_sc_corr_forest_fcs_calibrated_skeleton_kmeans_nearest_r12_two_way_subsampling.png` | `figure_code/figure_fcv_fcs_fu_sc_corr_forest.py` | Figure 9/12 summaries and compressed two-way-subsampling iterations | Matched analysis and figure |
| Fig. 5 | `figure_stimulus_delta_fcv_acd_combined_skeleton_kmeans_nearest_r12_two_way_subsampling.png` | `figure_code/figure_stimulus_delta_fcv_acd_combined_skeleton_kmeans_nearest_r12_two_way_subsampling.py` | Stimulus summaries, r12 structural summaries, and compressed two-way-subsampling iterations | Matched analysis and figure |
| Fig. 6 | `figure13_final_v2.png` | `figure_code/figure13_final_v2.py` | `derived_data/figure13/`, `raw_data/figure13/`, and `raw_data/figure_supply_15/figure13_inputs/` | Layer-model summaries plus frozen whole-brain simulation results |
| Fig. 7 | `figure_invertebrate_oo_fcv_relationships_with_137_subunits.png` | `figure_code/figure_invertebrate_oo_fcv_relationships_with_137_subunits.py` | `derived_data/invertebrates/` | Cellular and 137-unit fly analyses plus worm summaries |

## Supplementary-figure map

Supplementary numbering can change during final typesetting, so the stable
output filenames are used below.

| Output | Plotting code | Primary bundled input | Reproduction level |
|---|---|---|---|
| `figure_supply_1.png` | `figure_code/figure_supply_1.py` | `raw_data/figure_supply_1/latest_asset/` | Frozen six-panel QC asset; microscopy-scale MAT inputs are external |
| `figure_supply_2_proc.png` | `figure_code/figure_supply_2_proc.py` | `derived_data/figure12/` | Analysis-level table to figure |
| `figure_supply_5.png` | `figure_code/figure_supply_5.py` | `raw_data/figure_supply_5/` | Fixed network assets and null summaries to composite figure |
| `figure_supply_10_proc.png` | `figure_code/figure_supply_10_proc.py` | `derived_data/figure9/` | Analysis-level table to figure |
| `figure_supply_13.png` | `figure_code/figure_supply_13.py` | `derived_data/figure_supply_13/` | Frozen TE control summaries to figure |
| `figure_supply_15.png` | `figure_code/figure_supply_15.py` | `derived_data/figure13/` and `raw_data/figure_supply_15/` | Model summaries and representative traces to figure |
| `figure_supply_0.png` | `figure_code/figure_supply_0.py` | `raw_data/figure_supply_0/` | Frozen empirical and whole-brain simulation inputs to figure |
| `figure_supplement_te_structural_controls.png` | `figure_code/figure_supplement_te_structural_controls.py` | `derived_data/figure9/` and `derived_data/figure12/` | Matched region-level analysis and figure |
| `figure_stimulus_condition_region_profiles.png` | `figure_code/figure_stimulus_condition_region_profiles.py` | `derived_data/figure_stimulus/` | Subject-condition-region table to figure |
| `figure_invertebrate_anatomical_group_summary.png` | `figure_code/figure_invertebrate_anatomical_group_summary.py` | `derived_data/invertebrates/` | Node/recording summaries to figure |
| `figure_invertebrate_multiscale_sc_fc_matrices.png` | `figure_code/figure_invertebrate_multiscale_sc_fc_matrices.py` | `derived_data/invertebrates/invertebrate_multiscale_sc_fc_matrices.npz` | Compact multiscale matrix bundle to figure |

## Upstream data sources

The compact inputs were derived from the following experimental and
connectomic resources:

- Zebrafish whole-brain calcium imaging and atlas registration: [Chen et al.
  (2018)](https://doi.org/10.1016/j.neuron.2018.09.042).
- Cellular-resolution larval zebrafish anatomical reconstruction: [Kunst et
  al. (2019)](https://doi.org/10.1016/j.neuron.2019.04.034).
- Adult hermaphrodite *C. elegans* chemical connectome: [Cook et al.
  (2019)](https://doi.org/10.1038/s41586-019-1352-7), distributed through the
  WormWiring/OpenWorm resources.
- *C. elegans* whole-brain calcium recordings: the WormWideWeb dataset from
  [Atanas et al. (2023)](https://doi.org/10.1016/j.cell.2023.07.035).
- Adult female *Drosophila* structural connectivity: FlyWire783 annotations
  and connectome data described by [Dorkenwald et al.
  (2024)](https://doi.org/10.1038/s41586-024-07558-y).
- *Drosophila* functional recordings: the Branson999 calcium-imaging dataset
  from [Turner, Mann, and Clandinin
  (2021)](https://doi.org/10.1016/j.cub.2021.03.004); associated data are
  available from [figshare](https://doi.org/10.6084/m9.figshare.13349282.v3).

The large primary datasets are not redistributed in this repository. The
bundled tables are sufficient for the figure-level analyses documented here;
reconstruction from primary microscopy or cell-level connectivity requires
obtaining the original datasets from their providers. `DATA_SOURCES.md` and
`source_manifest.csv` describe how each bundled data group was derived and
where the public-release workflow begins.

## Reproducibility levels

This release distinguishes four levels of reproduction:

1. **Prepared-raw reproduction:** derived tables are recalculated from
   cell-, functional-unit-, ROI-, or edge-level prepared inputs using the
   scripts in `data_processing_code/`.
2. **Analysis-level reproduction:** statistical comparisons and plots are
   recalculated from bundled recording-, subject-, node-, or region-level
   tables.
3. **Summary-level reproduction:** plots and reported summaries are rebuilt
   from compact simulation or control-analysis outputs, without rerunning the
   large upstream computation.
4. **Asset-level reproduction:** a composite figure is reconstructed from
   fixed image assets and compact statistics. This applies to Supply 5.

The level assigned to each figure is shown in the tables above.

## Reproducibility boundary

The release provides prepared-raw-to-derived code for the zebrafish functional,
structural, and stimulus analyses; the invertebrate analyses; the compact layer
model; and selected supplementary controls. It is not an end-to-end
reconstruction from microscope-native files. The large zebrafish whole-brain
simulations remain represented by frozen observation tables, and Supply 5 uses
fixed network-diagram assets. See `RAW_DATA.md` and `DATA_SOURCES.md` for the
boundary of each data group.

The canonical zebrafish structure-function analyses use the same 42 anatomical
regions (the selected set excludes `rOB`), as recorded in
`derived_data/common/legacy_stimulus_forest_42_regions_no_rOB.csv`.
