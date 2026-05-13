#!/usr/bin/env python3
"""Expand adapter sweep configs into finetune_adapters.py commands."""

import argparse
import itertools
import json
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS_PATH = REPO_ROOT / "configs" / "pseudobulks.txt"
N_FOLDS = 5


DEFAULT_SWEEP = {
    "targets": ["rconvs", "fconv", "linear", "heads_only", "rconvs+heads", "all"],
    "conv_rank": [1, 2, 4, 8, 16],
    "linear_rank": [1, 2, 4, 8],
    "alpha": [1, 4, 16],
    "dropout": [0.0, 0.05],
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [0.0],
    "da_fraction": [0.5, 0.8],
    "epochs": 100,
    "patience": 10,
    "batch_size": 64,
    "jitter": 200,
    "loss_alpha": 100,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sweep-prefix", default="adapter")
    parser.add_argument("--pseudobulk", action="append", default=None)
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--materialize-model", action="store_true")
    args = parser.parse_args()

    sweep = _load_sweep(args.config)
    pseudobulks = args.pseudobulk
    if pseudobulks is None:
        pseudobulks = [p for p in PSEUDOBULKS_PATH.read_text().splitlines() if p.strip()]
    folds = args.fold if args.fold is not None else list(range(N_FOLDS))

    script = REPO_ROOT / "src" / "chrombpnet" / "da_finetuning" / "finetune_adapters.py"
    keys = ["targets", "conv_rank", "linear_rank", "alpha", "dropout", "lr", "weight_decay", "da_fraction"]
    combos = itertools.product(*[_as_list(sweep[key]) for key in keys])

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
                    "python",
                    str(script),
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
                    "--jitter",
                    str(sweep["jitter"]),
                    "--alpha",
                    str(sweep["loss_alpha"]),
                    "-v",
                ]
                if args.materialize_model:
                    cmd.append("--materialize-model")
                print(" ".join(shlex.quote(part) for part in cmd))


if __name__ == "__main__":
    main()
