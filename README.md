# Manuscript figure reproduction bundle

This directory contains the plotting code and compact, frozen analysis inputs
needed to reproduce six main-text figures and nine supplementary figures. It
does not contain the large calcium-imaging, cell-level connectome, or simulation
source datasets used upstream. Those data are represented here by documented
region-level tables, compact simulation summaries, or fixed image assets.

## Rebuild all figures

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all_figures.py
python validate_release.py
```

PNG outputs are written to `figures/` at 300 dpi. Statistical tables generated
by the plotting scripts are written to `statistics/`.
Generated figures, statistics, and intermediate plotting tables are ignored by
Git; run the commands above to recreate them locally. `checksums.sha256`
records the integrity of the distributed code and frozen inputs.

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
python figure_code/figure13_final_v2.py
python figure_code/figure_supply_15.py
```

To verify the distributed files before running the analysis, use:

```bash
sha256sum -c checksums.sha256
```

## Main-figure map

The internal filenames retain their development names. The manuscript mapping
is as follows.

| Manuscript figure | Output | Plotting code | Primary bundled input | Reproduction level |
|---|---|---|---|---|
| Fig. 2 | `figure9_final_v2.png` | `figure_code/figure9_final_v2.py` | `derived_data/figure9/` | Analysis-level tables to figure |
| Fig. 3 | `figure12_final_v2.png` | `figure_code/figure12_final_v2.py` | `derived_data/figure12/` | Analysis-level tables to figure |
| Fig. 4 | `figure_fcv_fcs_sc_corr_forest.png` | `figure_code/figure_fcv_fcs_sc_corr_forest.py` | `derived_data/figure9/`, `derived_data/figure12/`, and `derived_data/common/` | Matched region-level analysis and figure |
| Fig. 5 | `figure_stimulus_delta_fcv_acd_combined.png` | `figure_code/figure_stimulus_delta_fcv_acd_combined.py` | `derived_data/common/` | Frozen matched region-level summaries to figure |
| Fig. 6 | `figure13_final_v2.png` | `figure_code/figure13_final_v2.py` | `derived_data/figure13/`, `raw_data/figure13/`, and `raw_data/figure_supply_15/figure13_inputs/` | Layer-model summaries plus frozen whole-brain simulation results |
| Fig. 7 | `figure_invertebrate_oo_fcv_relationships.png` | `figure_code/figure_invertebrate_oo_fcv_relationships.py` | `derived_data/invertebrates/` | Matched node/region-level analysis and figure |

## Supplementary-figure map

Supplementary numbering can change during final typesetting, so the stable
output filenames are used below.

| Output | Plotting code | Primary bundled input | Reproduction level |
|---|---|---|---|
| `figure_supply_1.png` | `figure_code/figure_supply_1.py` | `raw_data/figure_supply_1/` | Compact summary to figure |
| `figure_supply_2_proc.png` | `figure_code/figure_supply_2_proc.py` | `derived_data/figure12/` | Analysis-level table to figure |
| `figure_supply_5.png` | `figure_code/figure_supply_5.py` | `raw_data/figure_supply_5/` | Fixed network assets and null summaries to composite figure |
| `figure_supply_10_proc.png` | `figure_code/figure_supply_10_proc.py` | `derived_data/figure9/` | Analysis-level table to figure |
| `figure_supply_13.png` | `figure_code/figure_supply_13.py` | `derived_data/figure_supply_13/` | Frozen TE control summaries to figure |
| `figure_supply_15.png` | `figure_code/figure_supply_15.py` | `derived_data/figure13/` and `raw_data/figure_supply_15/` | Model summaries and representative traces to figure |
| `figure_supplement_te_structural_controls.png` | `figure_code/figure_supplement_te_structural_controls.py` | `derived_data/figure9/` and `derived_data/figure12/` | Matched region-level analysis and figure |
| `figure_stimulus_condition_region_profiles.png` | `figure_code/figure_stimulus_condition_region_profiles.py` | `derived_data/figure_stimulus/` | Subject-condition-region table to figure |
| `figure_invertebrate_anatomical_group_summary.png` | `figure_code/figure_invertebrate_anatomical_group_summary.py` | `derived_data/invertebrates/` | Node/recording summaries to figure |

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

This release distinguishes three levels of reproduction:

1. **Analysis-level reproduction:** statistical comparisons and plots are
   recalculated from bundled recording-, subject-, node-, or region-level
   tables.
2. **Summary-level reproduction:** plots and reported summaries are rebuilt
   from compact simulation or control-analysis outputs, without rerunning the
   large upstream computation.
3. **Asset-level reproduction:** a composite figure is reconstructed from
   fixed image assets and compact statistics. This applies to Supply 5.

The level assigned to each figure is shown in the tables above.

## Reproducibility boundary

The release reproduces manuscript figures from final analysis-level inputs. It
is not an end-to-end reconstruction from raw microscopy or connectome files.
The four-layer stochastic model code is included because it is compact. The
large zebrafish whole-brain simulations are represented by their frozen
observation table, and Supply 5 uses fixed network-diagram assets. See
`DATA_SOURCES.md` for the provenance and interpretation of each data group.

The canonical zebrafish structure-function analyses use the same 42 anatomical
regions and exclude `rOB`, as recorded in
`derived_data/common/legacy_stimulus_forest_42_regions_no_rOB.csv`.
