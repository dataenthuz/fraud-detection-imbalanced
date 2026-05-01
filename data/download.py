"""
Download the Credit Card Fraud Detection dataset from Kaggle.

Setup:
  1. pip install kaggle
  2. kaggle.com > Account > API > Create New API Token
  3. Save kaggle.json to ~/.kaggle/kaggle.json (Windows: C:\\Users\\YOU\\.kaggle\\kaggle.json)
"""

import subprocess
import sys
from pathlib import Path

DATASET = "mlg-ulb/creditcardfraud"
DATA_DIR = Path(__file__).parent
EXPECTED = DATA_DIR / "creditcard.csv"


def main():
    if EXPECTED.exists():
        size_mb = EXPECTED.stat().st_size / 1e6
        print(f"Already downloaded: {EXPECTED} ({size_mb:.0f} MB)")
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Kaggle credentials not found.")
        print(f"  Expected: {kaggle_json}")
        print("  Steps:")
        print("    pip install kaggle")
        print("    Go to kaggle.com > Account > API > Create New API Token")
        print("    Save kaggle.json to ~/.kaggle/kaggle.json")
        sys.exit(1)

    print(f"Downloading {DATASET} to {DATA_DIR} ...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    size_mb = EXPECTED.stat().st_size / 1e6
    print(f"Done. {EXPECTED} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
