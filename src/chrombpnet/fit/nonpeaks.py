#!/usr/bin/env python3
"""Submit ChromBPNet nonpeak-generation jobs to SLURM."""

import argparse
import re
import sys

from slurm_launcher import (
    REPO_ROOT,
    SCRIPT_DIR,
    add_common_args,
    cleanup_then_run,
    resolve_dataset_and_jobs,
    submit_jobs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        default_mem="16G",
        default_time="12:00:00",
        default_gpus=0,
    )
    parser.add_argument(
        "-npr",
        "--neg-to-pos-ratio-train",
        type=int,
        default=1,
        help="Negatives per positive peak to generate for training folds (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pseudobulks, folds = resolve_dataset_and_jobs(args)
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    nonpeaks_dir = REPO_ROOT / "data" / f"nonpeaks{dataset_suffix}"
    nonpeaks_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / "logs" / "chrombpnet_nonpeaks"

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        for fold in folds:
            out_prefix = nonpeaks_dir / f"{pseudobulk}_fold_{fold}"
            done_path = nonpeaks_dir / f"{pseudobulk}_fold_{fold}_negatives.bed"
            if not args.rerun and done_path.exists():
                skipped += 1
                continue

            job_name = re.sub(
                r"[^A-Za-z0-9_]",
                "_",
                f"nonpeaks_{args.dataset}_{pseudobulk}_f{fold}".replace(".", "p"),
            )
            command = ["apptainer", "exec"]
            if args.gpus > 0:
                command.append("--nv")
            command.extend(
                [
                    "--bind",
                    f"{REPO_ROOT}:{REPO_ROOT}",
                    "--pwd",
                    SCRIPT_DIR,
                    args.apptainer_image,
                    "python",
                    "cli.py",
                    "prep",
                    "nonpeaks",
                    "-g",
                    REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa",
                    "-o",
                    out_prefix,
                    "-p",
                    REPO_ROOT
                    / "data"
                    / f"peaks{dataset_suffix}"
                    / f"{pseudobulk}_peaks.narrowPeak",
                    "-c",
                    REPO_ROOT / "data" / "Sbicolor_730_v5.0.chrom.sizes",
                    "-fl",
                    REPO_ROOT / "configs" / f"fold_{fold}.json",
                    "-npr",
                    args.neg_to_pos_ratio_train,
                ]
            )
            command = cleanup_then_run(
                [
                    f"{out_prefix}_auxiliary",
                    done_path,
                ],
                command,
            )
            jobs.append((job_name, command))

    submit_jobs(args=args, log_dir=log_dir, jobs=jobs, label="nonpeak")
    print(f"Skipped {skipped} completed nonpeak job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
