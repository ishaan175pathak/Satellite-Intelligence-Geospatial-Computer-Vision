"""
Download the WHU Building Dataset from Hugging Face.

Dataset: https://huggingface.co/datasets/giswqs/WHU-Building-Dataset
License: CC-BY-4.0

Setup:
    pip install huggingface_hub

Usage (from the project root):
    py src\\data\\download_dataset.py              # train split only (default)
    py src\\data\\download_dataset.py train val    # chosen splits
    py src\\data\\download_dataset.py all          # everything in the repo
"""

from pathlib import Path
import sys

from huggingface_hub import snapshot_download


REPO_ID = "giswqs/WHU-Building-Dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "whu_building"


def download(splits: list[str]) -> None:
    """Download selected splits (or everything). Resumes if interrupted."""

    patterns = None if splits == ["all"] else [f"{s}/**" for s in splits]

    print(f"[DOWNLOAD] {REPO_ID} -> {OUTPUT_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=OUTPUT_DIR,
        allow_patterns=patterns,
    )
    print("[DONE] Download complete.")


def validate() -> None:
    """Check that every image has a matching mask in each downloaded split."""

    split_dirs = [p for p in OUTPUT_DIR.iterdir() if (p / "Image").is_dir()]

    if not split_dirs:
        raise RuntimeError(
            f"No <split>/Image folders found in {OUTPUT_DIR}. "
            "Check the dataset layout on Hugging Face."
        )

    for split in sorted(split_dirs):
        images = {p.stem for p in (split / "Image").iterdir() if p.is_file()}
        masks = {p.stem for p in (split / "Mask").iterdir() if p.is_file()}
        unmatched = images ^ masks

        print(f"[VALIDATION] {split.name}: {len(images)} images, {len(masks)} masks")
        if unmatched:
            print(f"  [WARN] {len(unmatched)} files without a pair, e.g. {sorted(unmatched)[:3]}")


def inspect_georeferencing() -> None:
    """Report whether tiles carry CRS/geotransform (matters for Phase 4)."""

    try:
        import rasterio
    except ImportError:
        print("[INFO] rasterio not installed; skipping georeferencing check.")
        return

    sample = next(OUTPUT_DIR.glob("*/Image/*"), None)
    if sample is None:
        return

    try:
        with rasterio.open(sample) as src:
            print(f"[GEO] {sample.name}: size={src.width}x{src.height}, "
                  f"bands={src.count}, crs={src.crs}")
            print("[GEO] Georeferenced." if src.crs else
                  "[GEO] No CRS: polygons will be in pixel coordinates.")
    except Exception as exc:
        print(f"[GEO] Could not read {sample.name} with rasterio: {exc}")


def main() -> None:
    splits = sys.argv[1:] or ["train"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WHU BUILDING DATASET (Hugging Face)")
    print("=" * 60)

    download(splits)
    validate()
    inspect_georeferencing()

    print("=" * 60)
    print(f"DATASET READY: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()