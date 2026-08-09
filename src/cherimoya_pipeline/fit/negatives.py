#!/usr/bin/env python3
"""Submit Cherimoya GC-matched negative-region jobs to SLURM.

Negatives are sampled genome-wide per pseudobulk (not per fold): Cherimoya's
own `negatives` CLI has no chromosome-split concept, and PeakGenerator/
extract_loci already restrict extraction to the active fold's chromosomes at
train time, so one negatives BED per pseudobulk covers every fold.

By default, jobs run via `uv run --frozen` inside the shared plant-design
environment. Pass --apptainer to instead run via `apptainer exec --nv`
against the Cherimoya Apptainer image (see ../apptainer/), for clusters
whose system toolchain can't build/resolve cherimoya's
torch>=2.9/triton>=3.5.1 requirement.
"""

import argparse
import re
import sys

from slurm_launcher import (
    REPO_ROOT,
    add_common_args,
    cleanup_then_run,
    resolve_dataset_and_jobs,
    resolve_run_prefix,
    submit_jobs,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        default_mem="16G",
        default_time="4:00:00",
        default_gpus=0,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    species, pseudobulks, _folds = resolve_dataset_and_jobs(args)
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    data_dir = species_config.data_dir(species)
    negatives_dir = data_dir / f"negatives{dataset_suffix}"
    negatives_dir.mkdir(parents=True, exist_ok=True)
    log_dir = species_config.logs_dir(species) / "cherimoya_negatives"

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        out_path = negatives_dir / f"{pseudobulk}_negatives.bed"
        if not args.rerun and out_path.exists():
            skipped += 1
            continue

        job_name = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            f"negatives_{args.dataset}_{pseudobulk}".replace(".", "p"),
        )
        command = [
            *resolve_run_prefix(args),
            "cherimoya",
            "negatives",
            "-i",
            data_dir / f"peaks{dataset_suffix}" / f"{pseudobulk}_peaks.narrowPeak",
            "-o",
            out_path,
            "-f",
            species_config.genome_fasta(species),
            "-b",
            data_dir / f"bigwigs{dataset_suffix}" / f"{pseudobulk}.bw",
            "-v",
        ]
        command = cleanup_then_run([out_path], command)
        jobs.append((job_name, command))

    submit_jobs(args=args, log_dir=log_dir, jobs=jobs, label="negatives")
    print(f"Skipped {skipped} completed negatives job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
