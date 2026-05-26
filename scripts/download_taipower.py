#!/usr/bin/env python3
"""
Download Taipower historical generation data (dataset #37331).

Downloads the ~185MB JSON file containing 10-min generation data for all
Taipower-owned units over a rolling 3-month window.

Usage:
    python scripts/download_taipower.py
    python scripts/download_taipower.py --output data/raw/taipower_37331.json
"""

import argparse
import json
import sys
from pathlib import Path

import urllib.request

TAIPOWER_URL = (
    "https://service.taipower.com.tw/data/opendata/apply/file/d006010/001.json"
)
DEFAULT_OUTPUT = Path("data/raw/taipower_37331.json")

def download_taipower(output_path: Path, verbose: bool = True) -> None:
    """Download Taipower historical generation JSON.

    Args:
        output_path: Where to save the file.
        verbose: Print progress.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading Taipower dataset #37331...")
        print(f"  URL: {TAIPOWER_URL}")
        print(f"  Output: {output_path}")

    # Download with progress
    def _progress(block_num, block_size, total_size):
        if verbose and total_size > 0:
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    tmp_path = output_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(TAIPOWER_URL, tmp_path, reporthook=_progress)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(f"\nDownload failed: {e}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print()  # newline after progress

    # Validate JSON structure
    if verbose:
        print("  Validating JSON structure...")

    size_mb = tmp_path.stat().st_size / 1e6
    if size_mb < 10:
        tmp_path.unlink(missing_ok=True)
        print(f"  ERROR: File too small ({size_mb:.1f} MB), expected ~185 MB", file=sys.stderr)
        sys.exit(1)

    # Read with BOM handling
    raw_bytes = tmp_path.read_bytes()
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        raw_text = raw_bytes[3:].decode("utf-8")
    else:
        raw_text = raw_bytes.decode("utf-8")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        tmp_path.unlink(missing_ok=True)
        print(f"  ERROR: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Basic structure check
    top_key = next(iter(data.keys()), None)
    if top_key is None or "dataset" not in data[top_key]:
        tmp_path.unlink(missing_ok=True)
        print("  ERROR: Unexpected JSON structure (missing 'dataset' key)", file=sys.stderr)
        sys.exit(1)

    dataset = data[top_key]["dataset"]
    n_days = len(dataset)

    # Count renewable plants
    if n_days > 0:
        first_day = dataset[0]["data"]
        solar_names = {u["name"] for u in first_day if u.get("fueltype") == "太陽能"}
        wind_names = {u["name"] for u in first_day if u.get("fueltype") == "風力"}
    else:
        solar_names, wind_names = set(), set()

    # Move temp to final
    tmp_path.rename(output_path)

    if verbose:
        print(f"  File size: {size_mb:.1f} MB")
        print(f"  Days in dataset: {n_days}")
        print(f"  Date range: {dataset[0]['date']} to {dataset[-1]['date']}" if n_days > 0 else "  (empty)")
        print(f"  Solar plants: {len(solar_names)}")
        print(f"  Wind plants: {len(wind_names)}")
        print(f"  Download complete: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download Taipower generation data")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    download_taipower(Path(args.output), verbose=not args.quiet)

if __name__ == "__main__":
    main()
