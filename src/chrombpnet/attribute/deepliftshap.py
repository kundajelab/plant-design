"""Compute DeepLIFT/SHAP attributions for a trained BPNet model across all folds.

Loads trained models for each fold, computes hypothetical attributions on all
peaks genome-wide, averages across folds, and saves to attributions/bpnet/.

Usage:
    python src/bpnet/attribute/deepliftshap.py -p epidermis
"""

import argparse
import gc
import json
import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from bpnetlite.attribute import deep_lift_shap
from bpnetlite.bpnet import CountWrapper, ProfileWrapper
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--pseudobulk",
        type=str,
        required=True,
        choices=["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"],
        help="The name of the pseudobulk to benchmark",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        required=True,
        choices=["bias", "accessibility"],
        help="The name of the pseudobulk to benchmark",
    )
    parser.add_argument(
        "--head",
        type=str,
        required=True,
        choices=["profile", "count"],
        help="type of prediction to make (profile or count)",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Load chrom splits
    with open(CHROM_SPLITS_PATH, "r") as f:
        chrom_splits = json.load(f)
    n_folds = len(chrom_splits)

    # Construct data paths and verify that they exist.
    peaks_path = str(REPO_ROOT / f"data/peaks/{args.pseudobulk}_peaks_ACGT.narrowPeak")
    for path, label in [(peaks_path, "peaks")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Resolve model directory and verify that all models exist
    model_dir = REPO_ROOT / "models" / args.pseudobulk / "torch"
    model_paths = [
        model_dir / f"{args.pseudobulk}_f{fold}.{args.model_type}.torch"
        for fold in range(n_folds)
    ]
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    # Build params with defaults, then apply CLI overrides
    params = {
        "sequences": FASTA,
        "loci": peaks_path,
        "in_window": 2114,
        "batch_size": 64,
        "verbose": False,
    }
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.verbose:
        params["verbose"] = True

    # Load sequences
    all_chrom = list(chain.from_iterable(chrom_splits.values()))
    X = extract_loci(
        loci=params["loci"],
        sequences=params["sequences"],
        chroms=all_chrom,
        in_window=params["in_window"],
        verbose=params["verbose"],
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    )

    # Calculate attributions, looping over folds
    attributions = []
    for fold in range(n_folds):
        # Load model
        model = torch.load(
            model_paths[fold], weights_only=False, map_location=torch.device("cpu")
        )
        if args.head == "profile":
            model = ProfileWrapper(model)
        elif args.head == "count":
            model = CountWrapper(model)
        # Calculate attributions
        attributions.append(
            deep_lift_shap(
                model=model,
                X=X,
                verbose=params["verbose"],
                device="cuda",
                batch_size=params["batch_size"],
                warning_threshold=0.01,
                hypothetical=True,
            )
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Output directory
    out_dir = REPO_ROOT / "attributions"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save attributions
    out_path = out_dir / f"{args.pseudobulk}_{args.model_type}_{args.head}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, np.stack(attributions).mean(axis=0))
    print(f"\nAttributions saved to {out_path}")


if __name__ == "__main__":
    main()
