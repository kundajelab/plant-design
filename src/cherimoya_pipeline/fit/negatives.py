#!/usr/bin/env python3
"""Collect and print unfinished Cherimoya GC-matched negative-region jobs.

Negatives are sampled genome-wide per pseudobulk (not per fold): Cherimoya's
own `negatives` CLI has no chromosome-split concept, and PeakGenerator/
extract_loci already restrict extraction to the active fold's chromosomes at
train time, so one negatives BED per pseudobulk covers every fold.

Prints one shell command per pseudobulk with a missing negatives BED; pipe
into `parallel` or run directly.

Jobs run via `uv run --frozen` inside the shared plant-design environment.
"""

import argparse
import sys

from launcher_common import (
    add_common_args,
    cleanup_then_run,
    print_jobs,
    resolve_dataset_and_jobs,
)

import species_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    species, pseudobulks, _folds = resolve_dataset_and_jobs(args)
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    data_dir = species_config.data_dir(species)
    negatives_dir = data_dir / f"negatives{dataset_suffix}"
    negatives_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0
    for pseudobulk in pseudobulks:
        out_path = negatives_dir / f"{pseudobulk}_negatives.bed"
        if not args.rerun and out_path.exists():
            skipped += 1
            continue

        command = [
            "uv",
            "run",
            "--frozen",
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
        jobs.append((pseudobulk, command))

    print_jobs(jobs=jobs, label="negatives")
    print(f"Skipped {skipped} completed negatives job(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
