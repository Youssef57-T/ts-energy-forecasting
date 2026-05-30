"""Download PJM AEP hourly energy data from Kaggle."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def download_pjm_data(
    dataset: str = "robikscube/hourly-energy-consumption",
    region: str = "AEP",
    output_dir: Path | str = Path("data/raw"),
) -> Path:
    """Download and extract PJM hourly energy data for a specific region.

    Skips download if the target file already exists.

    Args:
        dataset: Kaggle dataset identifier.
        region: PJM region code (e.g. 'AEP', 'DEOK', 'DOM').
        output_dir: Directory to save raw data files.

    Returns:
        Path to the extracted CSV file.
    """
    import kaggle  # import here so missing install fails loudly at call-time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_file = output_dir / f"{region}_hourly.csv"
    if target_file.exists():
        print(f"Data already present at {target_file}, skipping download.")
        return target_file

    print(f"Downloading {dataset} from Kaggle...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=str(output_dir), unzip=True)

    if not target_file.exists():
        available = list(output_dir.glob("*.csv"))
        raise FileNotFoundError(
            f"Expected {target_file} not found after download. "
            f"Available CSVs: {available}"
        )

    print(f"Saved to {target_file}")
    return target_file


def compute_checksum(filepath: Path) -> str:
    """Return MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
