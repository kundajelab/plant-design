#!/usr/bin/env python3
"""Collect and print unfinished adapter sweep jobs, one per pseudobulk and fold.
For development/optimization of adapter hyperparameters.

Prints one shell command per job whose expected adapter bundle is missing;
pipe into `parallel`/`simple_gpu_scheduler` or run directly.
"""

import argparse
import itertools
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASETS = ("base", "fine", "coarse")
N_FOLDS = 5
PYTHON = ("uv", "run", "--frozen", "python")


DEFAULT_SWEEP = {
    "targets": [
        "all"
    ],  # ["rconvs", "fconv", "linear", "heads_only", "rconvs+heads", "all"],
    "conv_rank": [4, 8, 16],
    "linear_rank": [4, 8],
    "alpha": [4, 8, 16],
    "dropout": [0.0],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0],
    "da_fraction": [0.5, 0.8],
    "epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "max_jitter": 500,
    "negative_ratio": 0.25,
    "blacklist": None,
}


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _load_sweep(path):
    if path is None:
        return DEFAULT_SWEEP
    with open(path) as f:
        sweep = json.load(f)
    merged = DEFAULT_SWEEP.copy()
    merged.update(sweep)
    return merged


def model_root(species, dataset):
    root = species_config.models_dir(species)
    return root if dataset == "base" else root / dataset


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sweep-prefix", default="adapter")
    parser.add_argument("--pseudobulk", action="append", default=None)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to finetune (default: base).",
    )
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--materialize-model", action="store_true")
    parser.add_argument(
        "--check-finished",
        action="store_true",
        help="report expected adapter bundles and exit without printing commands",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="include commands whose expected adapter bundle already exists",
    )
    args = parser.parse_args()

    species = species_config.resolve_species(args.species)
    sweep = _load_sweep(args.config)
    pseudobulks = args.pseudobulk
    if pseudobulks is None:
        pseudobulks = species_config.read_pseudobulks(species, args.dataset)
    else:
        choices = species_config.read_pseudobulks(species, args.dataset)
        unknown = [p for p in pseudobulks if p not in choices]
        if unknown:
            print(
                f"Error: unknown pseudobulk(s) for species '{species}', dataset "
                f"'{args.dataset}': {', '.join(unknown)}. Choose one of: {', '.join(choices)}",
                file=sys.stderr,
            )
            sys.exit(1)
    folds = args.fold if args.fold is not None else list(range(N_FOLDS))

    script = REPO_ROOT / "src" / "cherimoya_pipeline" / "da_finetuning" / "finetune_adapters.py"
    keys = [
        "targets",
        "conv_rank",
        "linear_rank",
        "alpha",
        "dropout",
        "lr",
        "weight_decay",
        "da_fraction",
    ]
    combos = list(itertools.product(*[_as_list(sweep[key]) for key in keys]))
    commands_by_job = {
        (pseudobulk, fold): [] for pseudobulk in pseudobulks for fold in folds
    }

    for values in combos:
        combo = dict(zip(keys, values))
        target_tag = str(combo["targets"]).replace("+", "_")
        da_tag = str(combo["da_fraction"]).replace(".", "p")
        lr_tag = f"{combo['lr']:g}".replace(".", "p").replace("-", "m")
        sweep_id = (
            f"{args.sweep_prefix}_t-{target_tag}"
            f"_cr{combo['conv_rank']}_lr{combo['linear_rank']}"
            f"_a{combo['alpha']}_d{combo['dropout']:g}_lr{lr_tag}_da{da_tag}"
        )
        for pseudobulk in pseudobulks:
            for fold in folds:
                cmd = [
                    *PYTHON,
                    str(script),
                    "--species",
                    species,
                    "--dataset",
                    args.dataset,
                    "-p",
                    pseudobulk,
                    "-f",
                    str(fold),
                    "--sweep-id",
                    sweep_id,
                    "--targets",
                    str(combo["targets"]),
                    "--conv-rank",
                    str(combo["conv_rank"]),
                    "--linear-rank",
                    str(combo["linear_rank"]),
                    "--conv-alpha",
                    str(combo["alpha"]),
                    "--linear-alpha",
                    str(combo["alpha"]),
                    "--dropout",
                    str(combo["dropout"]),
                    "--lr",
                    str(combo["lr"]),
                    "--weight-decay",
                    str(combo["weight_decay"]),
                    "--da-fraction",
                    str(combo["da_fraction"]),
                    "--epochs",
                    str(sweep["epochs"]),
                    "--patience",
                    str(sweep["patience"]),
                    "--batch-size",
                    str(sweep["batch_size"]),
                    "--max-jitter",
                    str(sweep["max_jitter"]),
                    "--negative-ratio",
                    str(sweep["negative_ratio"]),
                    "-v",
                ]
                if sweep["blacklist"] is not None:
                    cmd.extend(["--blacklist", str(sweep["blacklist"])])
                if args.materialize_model:
                    cmd.append("--materialize-model")
                log_path = (
                    model_root(species, args.dataset)
                    / pseudobulk
                    / "adapters"
                    / sweep_id
                    / f"{pseudobulk}_f{fold}.log.tsv"
                )
                commands_by_job[(pseudobulk, fold)].append(
                    {
                        "command": " ".join(shlex.quote(part) for part in cmd),
                        "finish_path": log_path,
                        "sweep_id": sweep_id,
                    }
                )

    total_finished = 0
    total_expected = 0
    for (pseudobulk, fold), entries in commands_by_job.items():
        finished = sum(entry["finish_path"].exists() for entry in entries)
        total_finished += finished
        total_expected += len(entries)
        if args.check_finished:
            status = "DONE" if finished == len(entries) else "PENDING"
            print(
                f"{status}\t{pseudobulk}\tfold={fold}\t"
                f"{finished}/{len(entries)} training logs"
            )
            if finished != len(entries):
                for entry in entries:
                    if not entry["finish_path"].exists():
                        print(f"MISSING\t{entry['finish_path']}")

    if args.check_finished:
        print(f"\nFinished {total_finished}/{total_expected} expected training logs.")
        return

    printed = 0
    skipped = 0
    for entries in commands_by_job.values():
        for entry in entries:
            if args.rerun_finished or not entry["finish_path"].exists():
                print(entry["command"])
                printed += 1
            else:
                skipped += 1
    print(f"\nPrinted {printed} command(s); skipped {skipped} finished command(s).", file=sys.stderr)


if __name__ == "__main__":
    main()
