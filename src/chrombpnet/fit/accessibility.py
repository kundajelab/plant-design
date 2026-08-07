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

sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_mem="64G", default_time="72:00:00")
    return parser.parse_args()


def main():
    args = parse_args()
    species, pseudobulks, folds = resolve_dataset_and_jobs(args)
    log_dir = species_config.logs_dir(species) / "chrombpnet_accessibility"
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    data_dir = species_config.data_dir(species)
    model_root = species_config.models_dir(species)
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
                data_dir / f"bigwigs{dataset_suffix}" / f"{pseudobulk}.bw",
                "-d",
                "ATAC",
                "-g",
                species_config.genome_fasta(species),
                "-c",
                species_config.chrom_sizes(species),
                "-p",
                data_dir / f"peaks{dataset_suffix}" / f"{pseudobulk}_peaks.narrowPeak",
                "-n",
                data_dir
                / f"nonpeaks{dataset_suffix}"
                / f"{pseudobulk}_fold_{fold}_negatives.bed",
                "-fl",
                species_config.fold_path(species, fold),
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
