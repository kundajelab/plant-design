#!/usr/bin/env python3
"""Submit ChromBPNet accessibility-model training jobs to SLURM."""

import argparse

from slurm_launcher import (
    REPO_ROOT,
    add_common_args,
    resolve_dataset_and_jobs,
    sanitize_job_name,
    submit_jobs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_mem="64G", default_time="72:00:00")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, pseudobulks, folds = resolve_dataset_and_jobs(args)
    log_dir = REPO_ROOT / "logs" / "chrombpnet_accessibility"

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        for fold in folds:
            out_dir = dataset.model_dir(pseudobulk) / f"acc_f{fold}"
            nobias_done = out_dir / "models" / "chrombpnet_nobias.h5"
            bias_scaled_done = out_dir / "models" / "bias_model_scaled.h5"
            if not args.rerun and nobias_done.exists() and bias_scaled_done.exists():
                skipped += 1
                continue

            job_name = sanitize_job_name(f"acc_{args.dataset}_{pseudobulk}_f{fold}")
            command = [
                "python",
                "cli.py",
                "train",
                "-bw",
                dataset.bigwig_path(pseudobulk),
                "-d",
                "ATAC",
                "-g",
                REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa",
                "-c",
                REPO_ROOT / "data" / "Sbicolor_730_v5.0.chrom.sizes",
                "-p",
                dataset.peaks_path(pseudobulk),
                "-n",
                dataset.nonpeaks_path(pseudobulk, fold),
                "-fl",
                REPO_ROOT / "configs" / f"fold_{fold}.json",
                "-b",
                dataset.model_dir(pseudobulk)
                / f"bias_f{fold}"
                / "models"
                / f"f{fold}_bias.h5",
                "-o",
                out_dir,
            ]
            jobs.append((job_name, command))

    submit_jobs(args=args, log_dir=log_dir, jobs=jobs, label="accessibility")
    print(f"Skipped {skipped} completed accessibility job(s).")


if __name__ == "__main__":
    main()
