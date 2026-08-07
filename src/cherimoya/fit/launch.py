#!/usr/bin/env python3
"""Submit Cherimoya training jobs to SLURM.

Each job runs train_cherimoya.py for one pseudobulk/fold. Expects
negatives.py to have already produced a negatives BED for every pseudobulk
being trained.
"""

import argparse
import re
import sys

from slurm_launcher import (
    REPO_ROOT,
    add_common_args,
    cleanup_then_run,
    resolve_dataset_and_jobs,
    submit_jobs,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_mem="64G", default_time="24:00:00")
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    species, pseudobulks, folds = resolve_dataset_and_jobs(args)
    log_dir = species_config.logs_dir(species) / "cherimoya_fit"
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

            job_name = re.sub(
                r"[^A-Za-z0-9_]",
                "_",
                f"cherimoya_{args.dataset}_{pseudobulk}_f{fold}".replace(".", "p"),
            )
            command = [
                "uv",
                "run",
                "--frozen",
                "python",
                "train_cherimoya.py",
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
                "--early-stopping",
                args.early_stopping,
                "--negative-ratio",
                args.negative_ratio,
                "-o",
                out_dir,
            ]
            command = cleanup_then_run([done_path], command)
            jobs.append((job_name, command))

    submit_jobs(args=args, log_dir=log_dir, jobs=jobs, label="cherimoya fit")
    print(f"Skipped {skipped} completed fit job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
