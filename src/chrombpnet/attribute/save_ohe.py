"""
A small script to just extract and save OHE sequences for a given experiment.

Usage:
    python src/bpnet/attribute/save_ohe.py -p epidermis
"""

import argparse
import json
import sys
from itertools import chain
from pathlib import Path

import numpy as np
from da_peaks import (
    DATASETS,
    DA_PEAK_CHOICES,
    da_suffix,
    output_subdir,
    resolve_loci_path,
    resolve_ohe_path,
    validate_pseudobulk,
)
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--pseudobulk",
        type=str,
        required=True,
        help="The name of the pseudobulk to save OHE for.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to use (default: base).",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Save OHE sequences for a DA peak set instead of all cell-type peaks",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    try:
        validate_pseudobulk(REPO_ROOT, args.dataset, args.pseudobulk)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if args.da_peaks is not None and args.dataset != "base":
        print(
            "Error: --da-peaks is currently only defined for --dataset base.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Chromosome splits: fold i = test, fold (i+1)%n = validation, rest = train
    with open(CHROM_SPLITS_PATH, "r") as f:
        chrom_splits = json.load(f)

    # Construct data paths and verify that they exist.
    peaks_path = resolve_loci_path(
        REPO_ROOT, args.pseudobulk, args.da_peaks, dataset=args.dataset
    )
    for path, label in [(peaks_path, "peaks")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Build params with defaults, then apply CLI overrides
    params = {
        "sequences": FASTA,
        "loci": str(peaks_path),
        "in_window": 2114,
        "verbose": False,
    }
    if args.verbose:
        params["verbose"] = True

    # Load sequences to OHE
    all_chrom = list(chain.from_iterable(chrom_splits.values()))
    X = extract_loci(
        loci=params["loci"],
        sequences=params["sequences"],
        chroms=all_chrom,
        in_window=params["in_window"],
        verbose=params["verbose"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    )

    # Output directory
    out_dir = REPO_ROOT / "attributions" / output_subdir(args.dataset, "chrombpnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save OHE
    ohe_path = resolve_ohe_path(
        REPO_ROOT, args.pseudobulk, args.da_peaks, dataset=args.dataset
    )
    np.savez_compressed(ohe_path, X)
    print(f"\nOne-hot-encoded sequences{da_suffix(args.da_peaks)} saved to {ohe_path}")


if __name__ == "__main__":
    main()
