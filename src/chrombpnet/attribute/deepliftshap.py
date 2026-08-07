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
from da_peaks import (
    DATASETS,
    DA_PEAK_CHOICES,
    da_suffix,
    model_dir as dataset_model_dir,
    resolve_loci_path,
    validate_pseudobulk,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "chrombpnet" / "benchmark"))

import species_config  # noqa: E402
from daicon import (  # noqa: E402
    iter_adapters,
    load_accessibility_with_adapter,
)
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402


def assert_deeplift_shap_compatible(module):
    """Validate that daicon adapters are affine during DeepLIFT/SHAP."""
    for name, adapter in iter_adapters(module):
        dropout = getattr(adapter, "dropout", None)
        if isinstance(dropout, torch.nn.Dropout) and dropout.training:
            raise RuntimeError(
                f"Adapter '{name}' has active dropout. Call model.eval() before "
                "running DeepLIFT/SHAP."
            )
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "-p",
        "--pseudobulk",
        type=str,
        required=True,
        help="The name of the pseudobulk to benchmark",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to use (default: base).",
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
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run attributions on DA peaks instead of all cell-type peaks. "
             "'tau' uses da_peaks_tau.bed; 'pairwise' uses da_peaks_pairwise.bed; "
             "'pairwise_higher' uses per-cell-type directional pairwise BEDs; "
             "'specific_union' uses per-cell-type tau-specific OR pairwise-higher BEDs. "
             "Requires differential_accessibility.py and write_pairwise_higher_beds.py "
             "to have been run first.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned accessibility models instead of base torch/ models",
        adapter_help="Use default daicon adapter bundles for the selected DA fraction",
    )
    args = parser.parse_args()
    species = species_config.resolve_species(args.species)
    variant = resolve_model_variant(args)
    try:
        validate_pseudobulk(species, args.dataset, args.pseudobulk)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    fine_tag = variant["fine_tag"]
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]

    if args.finetuned and args.model_type == "bias":
        print(
            "Error: bias model is not saved during finetuning (bias is frozen). "
            "Use --model-type accessibility with --finetuned.",
            file=sys.stderr,
        )
        sys.exit(1)
    if adapter_id is not None and args.model_type == "bias":
        print(
            "Error: adapters are only defined for the accessibility model. "
            "Use --model-type accessibility with --adapter or --adapter-run-id.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.da_peaks is not None and args.dataset != "base":
        print(
            "Error: --da-peaks is currently only defined for --dataset base.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load chrom splits
    with open(species_config.config_dir(species) / "chrom_splits.json", "r") as f:
        chrom_splits = json.load(f)
    n_folds = len(chrom_splits)

    # Construct data paths and verify that they exist.
    if args.da_peaks is not None:
        peaks_path = resolve_loci_path(species, args.pseudobulk, args.da_peaks)
        if not peaks_path.exists():
            print(
                f"Error: {peaks_path.name} not found: {peaks_path}. "
                "Run differential_accessibility.py and write_pairwise_higher_beds.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        peaks_path = resolve_loci_path(
            species, args.pseudobulk, None, dataset=args.dataset
        )
        if not Path(peaks_path).exists():
            print(f"Error: peaks not found: {peaks_path}", file=sys.stderr)
            sys.exit(1)

    # Resolve model directory and verify that all models exist
    model_dir = dataset_model_dir(species, args.dataset, args.pseudobulk) / model_subdir
    model_paths = [
        model_dir / f"{args.pseudobulk}_f{fold}.{args.model_type}.torch"
        for fold in range(n_folds)
    ]
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)
    adapter_paths = [
        dataset_model_dir(species, args.dataset, args.pseudobulk)
        / "adapters"
        / adapter_id
        / f"{args.pseudobulk}_f{fold}.adapter.pt"
        for fold in range(n_folds)
    ] if adapter_id is not None else [None] * n_folds
    for adapter_path in adapter_paths:
        if adapter_path is not None and not adapter_path.exists():
            print(f"Error: adapter bundle not found: {adapter_path}", file=sys.stderr)
            sys.exit(1)

    # Build params with defaults, then apply CLI overrides
    params = {
        "sequences": str(species_config.genome_fasta(species)),
        "loci": str(peaks_path),
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
        if adapter_id is None:
            model = torch.load(
                model_paths[fold], weights_only=False, map_location=torch.device("cpu")
            )
        else:
            model, _manifest = load_accessibility_with_adapter(
                model_paths[fold],
                adapter_paths[fold],
                map_location=torch.device("cpu"),
            )
            model.eval()
            assert_deeplift_shap_compatible(model)
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
    if adapter_id is not None:
        base_subdir = f"chrombpnet_adapter_{adapter_id}"
    else:
        base_subdir = f"chrombpnet_{fine_tag}" if args.finetuned else "chrombpnet"
    out_subdir = base_subdir if args.dataset == "base" else f"{base_subdir}_{args.dataset}"
    out_dir = species_config.attributions_dir(species) / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save attributions
    out_path = (
        out_dir
        / f"{args.pseudobulk}_{args.model_type}_{args.head}{da_suffix(args.da_peaks)}.npz"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, np.stack(attributions).mean(axis=0))
    print(f"\nAttributions saved to {out_path}")


if __name__ == "__main__":
    main()
