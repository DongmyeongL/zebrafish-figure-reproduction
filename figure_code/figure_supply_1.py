"""Install the frozen Supplementary Figure 1 QC asset.

The microscopy-scale morphology and endpoint files needed to rebuild this
panel are not redistributed. The public release therefore provides the final
QC image and its fit statistics as an explicitly documented frozen asset.
"""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "raw_data" / "figure_supply_1" / "latest_asset"
OUT_PNG = ROOT / "figures" / "figure_supply_1.png"
OUT_STATS = ROOT / "statistics" / "figure_supply_1_stats.csv"


def main() -> None:
    source_png = ASSET_DIR / OUT_PNG.name
    source_stats = ASSET_DIR / OUT_STATS.name
    for source in (source_png, source_stats):
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen Figure S1 asset: {source}")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_png, OUT_PNG)
    shutil.copyfile(source_stats, OUT_STATS)
    print(f"Saved {OUT_PNG} (frozen QC asset)")
    print(f"Saved {OUT_STATS}")


if __name__ == "__main__":
    main()
