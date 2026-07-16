# Invertebrate processing pipeline

The scripts calculate FCV and directed structural measures directly from the
prepared species inputs documented in `../../RAW_DATA.md`.

Run either species from the repository root:

```bash
python scripts/run_pipeline.py \
  --target celegans \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/celegans/derived_data

python scripts/run_pipeline.py \
  --target drosophila \
  --raw-root /path/to/prepared_raw_data \
  --derived-root work/drosophila/derived_data
```

`01_compute_celegans_metrics.py` calculates FCV from spontaneous
WormWideWeb recording exports and calculates DCA, Pre/Post-DCA, and OO fraction
from the directed chemical-synapse edge list. The plotting analysis summarizes
matched neurons into eight literature-guided classes.

`02_compute_drosophila_metrics.py` calculates side-aware regional FCV from
Branson999 recording exports, maps FlyWire783 cells to dominant neuropils,
computes within-neuropil cell DCA, and summarizes inter-neuropil Pre/Post-DCA
and OO fraction. The matched output contains 41 side-aware regions.

The public validator checks output schema and sample coverage without loading a
private reference file:

```bash
python scripts/validate_derived.py celegans \
  --derived-root work/celegans/derived_data
python scripts/validate_derived.py drosophila \
  --derived-root work/drosophila/derived_data
```
