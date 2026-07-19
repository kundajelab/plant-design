#!/usr/bin/env python3
"""Submit ChromBPNet accessibility-model training jobs to SLURM."""

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
    add_common_args(parser, default_mem="64G", default_time="72:00:00")
    return parser.parse_args()


def main():
    args = parse_args()
    pseudobulks, folds = resolve_dataset_and_jobs(args)
    log_dir = REPO_ROOT / "logs" / "chrombpnet_accessibility"
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    model_root = REPO_ROOT / "models"
    if args.dataset != "base":
        model_root = model_root / args.dataset

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        for fold in folds:
            out_dir = model_root / pseudobulk / f"acc_f{fold}"
            nobias_done = out_dir / "models" / "chrombpnet_nobias.h5"
            bias_scaled_done = out_dir / "models" / "bias_model_scaled.h5"
            if not args.rerun and nobias_done.exists() and bias_scaled_done.exists():
                skipped += 1
                continue

            job_name = re.sub(
                r"[^A-Za-z0-9_]",
                "_",
                f"acc_{args.dataset}_{pseudobulk}_f{fold}".replace(".", "p"),
            )
            command = [
                "apptainer",
                "exec",
                "--nv",
                "--bind",
                f"{REPO_ROOT}:{REPO_ROOT}",
                "--pwd",
                SCRIPT_DIR,
                args.apptainer_image,
                "python",
                "cli.py",
                "train",
                "-bw",
                REPO_ROOT / "data" / f"bigwigs{dataset_suffix}" / f"{pseudobulk}.bw",
                "-d",
                "ATAC",
                "-g",
                REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa",
                "-c",
                REPO_ROOT / "data" / "Sbicolor_730_v5.0.chrom.sizes",
                "-p",
                REPO_ROOT
                / "data"
                / f"peaks{dataset_suffix}"
                / f"{pseudobulk}_peaks.narrowPeak",
                "-n",
                REPO_ROOT
                / "data"
                / f"nonpeaks{dataset_suffix}"
                / f"{pseudobulk}_fold_{fold}_negatives.bed",
                "-fl",
                REPO_ROOT / "configs" / f"fold_{fold}.json",
                "-b",
                model_root / pseudobulk / f"bias_f{fold}" / "models" / f"f{fold}_bias.h5",
                "-o",
                out_dir,
            ]
            command = cleanup_then_run([out_dir], command)
            jobs.append((job_name, command))

    submit_jobs(args=args, log_dir=log_dir, jobs=jobs, label="accessibility")
    print(f"Skipped {skipped} completed accessibility job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
