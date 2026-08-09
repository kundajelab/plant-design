#!/usr/bin/env python3
"""Collect and print unfinished Cherimoya training jobs.

Each job runs train_cherimoya.py for one pseudobulk/fold. Expects
negatives.py to have already produced a negatives BED for every pseudobulk
being trained. Prints one shell command per job with a missing model
checkpoint; pipe into `parallel`, `simple_gpu_scheduler`, or run directly.

Jobs run via `uv run --frozen` inside the shared plant-design environment.
"""

import argparse
import sys

from launcher_common import (
    SCRIPT_DIR,
    add_common_args,
    cleanup_then_run,
    print_jobs,
    resolve_dataset_and_jobs,
)

import species_config

TRAIN_SCRIPT = SCRIPT_DIR / "train_cherimoya.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop after this many epochs with no validation improvement "
        "(default: disabled -- train the full --max-epochs).",
    )
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    species, pseudobulks, folds = resolve_dataset_and_jobs(args)
    model_root = species_config.models_dir(species)
    if args.dataset != "base":
        model_root = model_root / args.dataset

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        for fold in folds:
            out_dir = model_root / pseudobulk
            done_path = out_dir / f"{pseudobulk}_f{fold}.torch"
            if not args.rerun and done_path.exists():
                skipped += 1
                continue

            command = [
                "uv",
                "run",
                "--frozen",
                "python",
                TRAIN_SCRIPT,
                "--species",
                species,
                "--dataset",
                args.dataset,
                "-p",
                pseudobulk,
                "-f",
                fold,
                "--n-filters",
                args.n_filters,
                "--n-layers",
                args.n_layers,
                "--max-epochs",
                args.max_epochs,
                "--negative-ratio",
                args.negative_ratio,
                "-o",
                out_dir,
            ]
            if args.early_stopping is not None:
                command += ["--early-stopping", args.early_stopping]
            command = cleanup_then_run([done_path], command)
            jobs.append((f"{pseudobulk}_f{fold}", command))

    print_jobs(jobs=jobs, label="cherimoya fit")
    print(f"Skipped {skipped} completed fit job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
