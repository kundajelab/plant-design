#!/usr/bin/env python3
"""Collect and print unfinished DeepLIFT/SHAP attribution jobs for Cherimoya models.

Reads pseudobulks from configs/<species>/pseudobulks.txt and prints one
shell command per (pseudobulk, head) pair via `deepliftshap.py` whose output
is missing; pipe into `parallel` or run directly.

Jobs are skipped if the attribution output already exists, if the required
loci BED file is missing (run filter_nonACGT_regions.py, differential_accessibility.py,
or write_pairwise_higher_beds.py as needed), or if the required model files are
missing (run train_cherimoya.py first).

Usage:
    python src/cherimoya_pipeline/attribute/launch.py                                    # all pseudobulks, both heads
    python src/cherimoya_pipeline/attribute/launch.py --head count                       # count head only
    python src/cherimoya_pipeline/attribute/launch.py --finetuned                        # finetuned models (50:50 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --finetuned --da-fraction 0.8     # finetuned models (80:20 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --adapter                         # default daicon adapters (50:50 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --adapter --da-fraction 0.8       # default daicon adapters (80:20 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --da-peaks tau                    # attributions on tau-specific peaks only
    python src/cherimoya_pipeline/attribute/launch.py --da-peaks specific_union       # attributions on per-cell-type union DA peaks
"""

import argparse
import sys
from pathlib import Path
from da_peaks import (
    DATASETS,
    DA_PEAK_CHOICES,
    da_suffix,
    model_dir as dataset_model_dir,
    read_pseudobulks,
    resolve_loci_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "cherimoya_pipeline" / "benchmark"))

import species_config  # noqa: E402
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402

N_FOLDS = 5


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution head(s) to run; repeatable (default: profile count)",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run attributions on DA peaks instead of all cell-type peaks (default: all peaks)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to use (default: base).",
    )
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned models instead of the base model",
        adapter_help="Use default daicon adapter bundles for the selected DA fraction",
    )
    # deepliftshap parameters
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="batch size for attribution (default: 64)",
    )
    args = parser.parse_args()
    species = species_config.resolve_species(args.species)
    variant = resolve_model_variant(args)
    fine_tag = variant["fine_tag"]
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]

    heads = args.head if args.head is not None else ["profile", "count"]

    if adapter_id is not None:
        base_subdir = f"cherimoya_adapter_{adapter_id}"
    else:
        base_subdir = f"cherimoya_{fine_tag}" if args.finetuned else "cherimoya"
    out_subdir = base_subdir if args.dataset == "base" else f"{base_subdir}_{args.dataset}"

    if args.da_peaks is not None and args.dataset != "base":
        print(
            "Error: --da-peaks is currently only defined for --dataset base.",
            file=sys.stderr,
        )
        sys.exit(1)

    pseudobulks = read_pseudobulks(species, args.dataset)

    attr_dir = species_config.attributions_dir(species)

    printed = 0
    skipped_done = 0
    skipped_missing = 0

    for pseudobulk in pseudobulks:
        peaks_path = resolve_loci_path(
            species, pseudobulk, args.da_peaks, dataset=args.dataset
        )
        if not peaks_path.exists():
            skipped_missing += len(heads)
            print(
                f"SKIP {pseudobulk}: loci file missing ({peaks_path})",
                file=sys.stderr,
            )
            continue

        model_dir = dataset_model_dir(species, args.dataset, pseudobulk) / model_subdir
        model_paths = [
            model_dir / f"{pseudobulk}_f{fold}.torch" for fold in range(N_FOLDS)
        ]
        missing_models = [p for p in model_paths if not p.exists()]
        if missing_models:
            skipped_missing += len(heads)
            print(
                f"SKIP {pseudobulk}: {len(missing_models)} model file(s) missing",
                file=sys.stderr,
            )
            continue
        if adapter_id is not None:
            adapter_paths = [
                dataset_model_dir(species, args.dataset, pseudobulk)
                / "adapters"
                / adapter_id
                / f"{pseudobulk}_f{fold}.adapter.pt"
                for fold in range(N_FOLDS)
            ]
            missing_adapters = [p for p in adapter_paths if not p.exists()]
            if missing_adapters:
                skipped_missing += len(heads)
                print(
                    f"SKIP {pseudobulk}: {len(missing_adapters)} adapter bundle(s) missing",
                    file=sys.stderr,
                )
                continue

        for head in heads:
            suffix = da_suffix(args.da_peaks)
            out_path = attr_dir / out_subdir / f"{pseudobulk}_{head}{suffix}.npz"
            if out_path.exists():
                skipped_done += 1
                continue

            script_path = Path(__file__).resolve().parent / "deepliftshap.py"

            finetuned_flags = (
                f" --finetuned --da-fraction {args.da_fraction}" if args.finetuned else ""
            )
            if adapter_id is None:
                adapter_flags = ""
            elif args.adapter:
                adapter_flags = f" --adapter --da-fraction {args.da_fraction}"
            else:
                adapter_flags = f" --adapter-sweep-id {adapter_id}"
            da_peaks_flag = f" --da-peaks {args.da_peaks}" if args.da_peaks is not None else ""
            deepliftshap_cmd = (
                f"uv run python {script_path}"
                f" --species {species}"
                f" -p {pseudobulk}"
                f" --dataset {args.dataset}"
                f" --head {head}"
                f" -b {args.batch_size}"
                f" -v"
                f"{finetuned_flags}"
                f"{adapter_flags}"
                f"{da_peaks_flag}"
            )

            print(deepliftshap_cmd)
            printed += 1

    total = len(pseudobulks) * len(heads)
    print(
        f"\nPrinted {printed} command(s), skipped {skipped_missing} missing inputs, "
        f"skipped {skipped_done} already done ({total} total)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
