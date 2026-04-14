"""
Download DIV2K training and validation HR images and run a quick sanity check.

Usage:
    python download_data.py            # download both sets + sanity check
    python download_data.py --check    # sanity check only (skip download)

Downloads:
    DIV2K_train_HR.zip  (~3.5 GB)  → data/DIV2K/DIV2K_train_HR/
    DIV2K_valid_HR.zip  (~430 MB)  → data/DIV2K/DIV2K_valid_HR/
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required: pip install pillow")


# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
DATA_DIR  = REPO_ROOT / "data" / "DIV2K"

DATASETS = [
    {
        "url":        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "zip":        "DIV2K_train_HR.zip",
        "subdir":     "DIV2K_train_HR",
        "expected_n": 800,
        "label":      "train",
    },
    {
        "url":        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "zip":        "DIV2K_valid_HR.zip",
        "subdir":     "DIV2K_valid_HR",
        "expected_n": 100,
        "label":      "validation",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct // 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  "
              f"({downloaded/1e9:.2f}/{total_size/1e9:.2f} GB)", end="", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"         → {dest}")
    urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress bar


def extract(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path.name} → {dest_dir}/")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if i % 100 == 0 or i == len(members):
                print(f"  {i}/{len(members)} files extracted", end="\r", flush=True)
    print()


# ── Download ──────────────────────────────────────────────────────────────────
def run_download_one(ds: dict) -> None:
    zip_path = DATA_DIR / ds["zip"]
    dest_dir = DATA_DIR / ds["subdir"]

    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"Already extracted: {dest_dir}  (delete it to re-download)")
        return

    if not zip_path.exists():
        download(ds["url"], zip_path)
    else:
        print(f"Zip already present: {zip_path}  (skipping download)")

    extract(zip_path, DATA_DIR)

    zip_path.unlink()
    print(f"Removed {zip_path.name}")


def run_download() -> None:
    for ds in DATASETS:
        print(f"\n── {ds['label'].upper()} set ──")
        run_download_one(ds)


# ── Sanity check ──────────────────────────────────────────────────────────────
def check_one(ds: dict) -> None:
    target_dir = DATA_DIR / ds["subdir"]

    if not target_dir.exists():
        print(f"WARNING: {target_dir} not found — skipping check for {ds['label']} set")
        return

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    images = sorted(
        p for p in target_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    print(f"\n── DIV2K {ds['label']} sanity check ──────────────────────")
    print(f"Directory : {target_dir}")
    print(f"Images    : {len(images)}  (expected {ds['expected_n']})")

    if len(images) != ds["expected_n"]:
        print(f"WARNING: expected {ds['expected_n']} images, found {len(images)}")

    if not images:
        print("No images to inspect.")
        return

    min_w = min_h = float("inf")
    max_w = max_h = 0
    min_img = max_img = None

    for p in images:
        try:
            w, h = Image.open(p).size
        except Exception as e:
            print(f"  Could not open {p.name}: {e}")
            continue
        if w * h < min_w * min_h:
            min_w, min_h, min_img = w, h, p.name
        if w * h > max_w * max_h:
            max_w, max_h, max_img = w, h, p.name

    print(f"Min res   : {min_w}×{min_h}  ({min_img})")
    print(f"Max res   : {max_w}×{max_h}  ({max_img})")
    print(f"────────────────────────────────────────────────────────")


def run_check() -> None:
    for ds in DATASETS:
        check_one(ds)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Only run the sanity check; skip downloading")
    args = parser.parse_args()

    if not args.check:
        run_download()

    run_check()
