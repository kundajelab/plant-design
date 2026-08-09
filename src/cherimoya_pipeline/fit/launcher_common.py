"""Shared helpers for Cherimoya job launchers.

No SLURM: launchers just collect unfinished jobs (missing output files) and
print one shell command per job, to be run directly or piped into
`parallel`/`simple_gpu_scheduler`/etc.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASET_NAMES = ("base", "fine", "coarse")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="base",
        help="Dataset namespace to train (default: base).",
    )
    parser.add_argument(
        "-p",
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        default=None,
        help="Pseudobulk to train; repeatable. Defaults to all pseudobulks in --dataset.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        dest="folds",
        type=int,
        action="append",
        choices=range(5),
        default=None,
        help="Fold to train; repeatable. Defaults to all folds.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Print jobs even when completion marker files already exist.",
    )


def validate_pseudobulk(species: str, dataset: str, pseudobulk: str) -> None:
    choices = species_config.read_pseudobulks(species, dataset)
    if pseudobulk not in choices:
        joined = ", ".join(choices)
        raise SystemExit(
            f"Error: unknown pseudobulk '{pseudobulk}' for species '{species}', "
            f"dataset '{dataset}'. Choose one of: {joined}"
        )


def resolve_dataset_and_jobs(args: argparse.Namespace):
    species = species_config.resolve_species(args.species)
    pseudobulks = args.pseudobulks or species_config.read_pseudobulks(
        species, args.dataset
    )
    folds = args.folds or list(range(5))
    for pseudobulk in pseudobulks:
        validate_pseudobulk(species, args.dataset, pseudobulk)
    return species, pseudobulks, folds


def cleanup_then_run(cleanup_paths: list[str | Path], command: list[str | Path]):
    cleanup = " ".join(shlex.quote(str(path)) for path in cleanup_paths)
    quoted_command = " ".join(shlex.quote(str(part)) for part in command)
    return ["bash", "-lc", f"rm -rf -- {cleanup} && {quoted_command}"]


def print_jobs(*, jobs: list[tuple[str, list[str | Path]]], label: str) -> None:
    """Print one shell command per unfinished job."""
    for _job_name, command in jobs:
        print(" ".join(shlex.quote(str(part)) for part in command))
    print(f"Printed {len(jobs)} {label} command(s).", file=sys.stderr)
