#!/usr/bin/env python3
"""Collect and print unfinished standard DA fine-tuning jobs.

This launcher is for the small set of named runs used downstream:

- full-model fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter fine-tuning at 0.5 and 0.8 DA fractions

Use ``launch_adapter_sweep.py`` for the full adapter hyperparameter sweep.

Prints one shell command per job whose expected training log is missing;
pipe into `parallel`/`simple_gpu_scheduler` or run directly.
"""

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASETS = ("base", "fine", "coarse")
N_FOLDS = 5
DA_FRACTIONS = (0.5, 0.8)
PYTHON = ("uv", "run", "--frozen", "python")

RUN_GROUPS = ("full", "adapter")
DEFAULT_FULL_LR = 1e-4
DEFAULT_ADAPTER_LR = 1e-3


def da_tag(da_fraction):
    return str(da_fraction).replace(".", "p")


def full_tag(da_fraction):
    return "finetuned" if da_fraction == 0.5 else f"finetuned_{da_fraction:g}"


def default_adapter_run_id(da_fraction):
    return f"daicon_default_da{da_tag(da_fraction)}"


def quote_command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def model_root(species, dataset):
    root = species_config.models_dir(species)
    return root if dataset == "base" else root / dataset


def load_pseudobulks(species, dataset, selected):
    if selected is not None:
        return selected
    return species_config.read_pseudobulks(species, dataset)


def common_training_args(args, lr):
    return [
        "--epochs",
        args.epochs,
        "--patience",
        args.patience,
        "--batch-size",
        args.batch_size,
        "--jitter",
        args.jitter,
        "--alpha",
        args.loss_alpha,
        "--lr",
        lr,
    ]


def build_full_job(args, species, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "cherimoya_pipeline" / "da_finetuning" / "finetune_da.py"
    command = [
        *PYTHON,
        script,
        "--species",
        species,
        "--dataset",
        args.dataset,
        "-p",
        pseudobulk,
        "-f",
        fold,
        "--mode",
        "full",
        "--da-fraction",
        da_fraction,
        *common_training_args(args, args.full_lr),
        "-v",
    ]
    finish_path = (
        model_root(species, args.dataset)
        / pseudobulk
        / full_tag(da_fraction)
        / f"{pseudobulk}_f{fold}.log"
    )
    return {
        "run_group": "full",
        "run_id": full_tag(da_fraction),
        "command": quote_command(command),
        "finish_path": finish_path,
    }


def build_adapter_job(args, species, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "cherimoya_pipeline" / "da_finetuning" / "finetune_da.py"
    run_id = default_adapter_run_id(da_fraction)
    command = [
        *PYTHON,
        script,
        "--species",
        species,
        "--dataset",
        args.dataset,
        "-p",
        pseudobulk,
        "-f",
        fold,
        "--mode",
        "adapter",
        "--da-fraction",
        da_fraction,
        *common_training_args(args, args.adapter_lr),
        "-v",
    ]
    finish_path = (
        model_root(species, args.dataset)
        / pseudobulk
        / "adapters"
        / run_id
        / f"{pseudobulk}_f{fold}.log.tsv"
    )
    return {
        "run_group": "adapter",
        "run_id": run_id,
        "command": quote_command(command),
        "finish_path": finish_path,
    }


def build_jobs(args, species):
    pseudobulks = load_pseudobulks(species, args.dataset, args.pseudobulk)
    folds = args.fold if args.fold is not None else list(range(N_FOLDS))
    da_fractions = args.da_fraction if args.da_fraction is not None else list(DA_FRACTIONS)

    builders = {
        "full": build_full_job,
        "adapter": build_adapter_job,
    }
    jobs = {}
    for run_group in args.run:
        for pseudobulk in pseudobulks:
            for fold in folds:
                key = (run_group, pseudobulk, fold)
                jobs[key] = [
                    builders[run_group](args, species, pseudobulk, fold, da_fraction)
                    for da_fraction in da_fractions
                ]
    return jobs


def print_finished_report(jobs):
    total_finished = 0
    total_expected = 0
    for (run_group, pseudobulk, fold), entries in jobs.items():
        finished = sum(entry["finish_path"].exists() for entry in entries)
        total_finished += finished
        total_expected += len(entries)
        status = "DONE" if finished == len(entries) else "PENDING"
        print(
            f"{status}\t{run_group}\t{pseudobulk}\tfold={fold}\t"
            f"{finished}/{len(entries)} training logs"
        )
        if finished != len(entries):
            for entry in entries:
                if not entry["finish_path"].exists():
                    print(f"MISSING\t{entry['finish_path']}")
    print(f"\nFinished {total_finished}/{total_expected} expected training logs.")


def print_jobs(args, jobs):
    printed = 0
    skipped = 0
    for entries in jobs.values():
        for entry in entries:
            if args.rerun_finished or not entry["finish_path"].exists():
                print(entry["command"])
                printed += 1
            else:
                skipped += 1
    print(f"\nPrinted {printed} command(s); skipped {skipped} finished command(s).", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--run",
        action="append",
        choices=RUN_GROUPS,
        default=None,
        help="Run group to submit; repeatable. Default: all groups.",
    )
    parser.add_argument("--pseudobulk", action="append", default=None)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to finetune (default: base).",
    )
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--da-fraction", type=float, action="append", default=None)
    parser.add_argument(
        "--full-lr",
        type=float,
        default=DEFAULT_FULL_LR,
        help=f"Learning rate for full-model fine-tuning (default: {DEFAULT_FULL_LR:g})",
    )
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=DEFAULT_ADAPTER_LR,
        help=f"Learning rate for adapter fine-tuning (default: {DEFAULT_ADAPTER_LR:g})",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jitter", type=int, default=200)
    parser.add_argument("--loss-alpha", type=int, default=100)
    parser.add_argument(
        "--check-finished",
        action="store_true",
        help="report expected training logs and exit",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="include commands whose expected training log already exists",
    )
    args = parser.parse_args()

    if args.run is None:
        args.run = list(RUN_GROUPS)
    species = species_config.resolve_species(args.species)
    choices = load_pseudobulks(species, args.dataset, None)
    selected = args.pseudobulk or choices
    unknown = [p for p in selected if p not in choices]
    if unknown:
        print(
            f"Error: unknown pseudobulk(s) for species '{species}', dataset "
            f"'{args.dataset}': {', '.join(unknown)}. Choose one of: {', '.join(choices)}",
            file=sys.stderr,
        )
        sys.exit(1)

    jobs = build_jobs(args, species)
    if args.check_finished:
        print_finished_report(jobs)
        return

    print_jobs(args, jobs)


if __name__ == "__main__":
    main()
