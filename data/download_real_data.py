"""
Real Data Downloader
=====================
Downloads the E-Commerce Shipping Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/prachi13/customer-analytics
License: Other (specified in description)
Records: 10,999 shipment records with on-time/delayed labels

Usage:
    python data/download_real_data.py
"""

import os
import sys
import zipfile
import shutil

# Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_CSV = os.path.join(RAW_DIR, "ecommerce_shipping.csv")

KAGGLE_DATASET = "prachi13/customer-analytics"
EXPECTED_FILENAME = "E Commerce Shipping Dataset.csv"


def download_via_kaggle_api() -> bool:
    """Download using the official Kaggle API (requires kaggle.json credentials)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        print("🔑 Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        print(f"⬇️  Downloading dataset: {KAGGLE_DATASET}")
        os.makedirs(RAW_DIR, exist_ok=True)
        api.dataset_download_files(KAGGLE_DATASET, path=RAW_DIR, unzip=True)

        # The dataset extracts as "E Commerce Shipping Dataset.csv"
        extracted_file = os.path.join(RAW_DIR, EXPECTED_FILENAME)
        if os.path.exists(extracted_file):
            # Rename to our standard name
            shutil.move(extracted_file, RAW_CSV)
            print(f"✅ Downloaded and saved: {RAW_CSV}")
            return True
        else:
            # Try to find any CSV in the raw dir
            for fname in os.listdir(RAW_DIR):
                if fname.endswith(".csv"):
                    src = os.path.join(RAW_DIR, fname)
                    shutil.move(src, RAW_CSV)
                    print(f"✅ Downloaded and saved: {RAW_CSV} (from {fname})")
                    return True

        print("❌ Download succeeded but CSV not found in extracted files.")
        return False

    except ImportError:
        print("⚠️  kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"⚠️  Kaggle API download failed: {e}")
        return False


def download_via_url() -> bool:
    """Fallback: download via direct URL using urllib."""
    try:
        import urllib.request
        import io

        print("⬇️  Attempting direct download (fallback)...")
        url = "https://www.kaggle.com/api/v1/datasets/download/prachi13/customer-analytics"

        os.makedirs(RAW_DIR, exist_ok=True)
        zip_path = os.path.join(RAW_DIR, "dataset.zip")

        print(f"   Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)

        # Extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        os.remove(zip_path)

        # Find and rename
        extracted_file = os.path.join(RAW_DIR, EXPECTED_FILENAME)
        if os.path.exists(extracted_file):
            shutil.move(extracted_file, RAW_CSV)
            print(f"✅ Downloaded and saved: {RAW_CSV}")
            return True
        else:
            for fname in os.listdir(RAW_DIR):
                if fname.endswith(".csv"):
                    src = os.path.join(RAW_DIR, fname)
                    shutil.move(src, RAW_CSV)
                    print(f"✅ Downloaded and saved: {RAW_CSV} (from {fname})")
                    return True

        print("❌ Download succeeded but CSV not found.")
        return False

    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        return False


def download_dataset() -> str:
    """
    Download the real dataset. Tries Kaggle API first, then falls back to URL.

    Returns:
        Path to the downloaded CSV, or None if all methods fail.
    """
    # If already downloaded, skip
    if os.path.exists(RAW_CSV):
        file_size = os.path.getsize(RAW_CSV)
        if file_size > 100_000:  # Should be ~500KB+
            print(f"📦 Dataset already exists: {RAW_CSV} ({file_size:,} bytes)")
            return RAW_CSV
        else:
            print(f"⚠️  Existing file too small ({file_size} bytes), re-downloading...")

    print("=" * 60)
    print("📥 DOWNLOADING REAL LOGISTICS DATASET")
    print("   Source: Kaggle E-Commerce Shipping Dataset")
    print("   Author: Prachi Gopalani")
    print("   Records: 10,999 shipments")
    print("=" * 60)

    # Method 1: Kaggle API
    if download_via_kaggle_api():
        return RAW_CSV

    # Method 2: Direct URL
    if download_via_url():
        return RAW_CSV

    # All methods failed
    print("\n" + "=" * 60)
    print("❌ AUTOMATIC DOWNLOAD FAILED")
    print("=" * 60)
    print("Please download manually:")
    print("  1. Go to: https://www.kaggle.com/datasets/prachi13/customer-analytics")
    print("  2. Click 'Download' button")
    print(f"  3. Extract CSV to: {RAW_CSV}")
    print("\nAlternatively, set up Kaggle credentials:")
    print("  1. Go to: https://www.kaggle.com/settings")
    print("  2. Click 'Create New Token' → saves kaggle.json")
    print("  3. Place kaggle.json in ~/.kaggle/")
    print("  4. Re-run this script")
    return None


if __name__ == "__main__":
    result = download_dataset()
    if result:
        import pandas as pd
        df = pd.read_csv(result)
        print(f"\n📊 Dataset preview:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n{df.head()}")
    else:
        sys.exit(1)
