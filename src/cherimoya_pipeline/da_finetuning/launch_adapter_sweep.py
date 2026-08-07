#!/usr/bin/env python3
"""Submit one SLURM adapter sweep job per pseudobulk and data fold.
For development/optimization of adapter hyperparameters."""

import argparse
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


def dataset_suffix(dataset):
    return "" if dataset == "base" else f"_{dataset}"


def model_root(dataset):
    root = REPO_ROOT / "models"
    return root if dataset == "base" else root / dataset


def read_pseudobulks(dataset):
    path = REPO_ROOT / "configs" / f"pseudobulks{dataset_suffix(dataset)}.txt"
    return [
        p.strip()
        for p in path.read_text().splitlines()
        if p.strip() and not p.startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
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
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="print expanded python commands to stdout instead of submitting sbatch jobs",
    )
    parser.add_argument(
        "--check-finished",
        action="store_true",
        help="report expected adapter bundles and exit without printing commands or submitting jobs",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="include commands whose expected adapter bundle already exists",
    )
    parser.add_argument(
        "--scheduler-gpus",
        nargs="+",
        default=None,
        help="GPU IDs passed to simple_gpu_scheduler inside each sbatch job; omit to run commands sequentially",
    )
    parser.add_argument("--partition", type=str, default="gpu,akundaje,owners")
    parser.add_argument(
        "--constraint", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR"
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2-00:00:00")
    parser.add_argument("--numba-threads", type=int, default=4)
    args = parser.parse_args()

    sweep = _load_sweep(args.config)
    pseudobulks = args.pseudobulk
    if pseudobulks is None:
        pseudobulks = read_pseudobulks(args.dataset)
    else:
        choices = read_pseudobulks(args.dataset)
        unknown = [p for p in pseudobulks if p not in choices]
        if unknown:
            print(
                f"Error: unknown pseudobulk(s) for dataset '{args.dataset}': "
                f"{', '.join(unknown)}. Choose one of: {', '.join(choices)}",
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
                    "--jitter",
                    str(sweep["jitter"]),
                    "--alpha",
                    str(sweep["loss_alpha"]),
                    "-v",
                ]
                if args.materialize_model:
                    cmd.append("--materialize-model")
                log_path = (
                    model_root(args.dataset)
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

    if args.interactive:
        for entries in commands_by_job.values():
            for entry in entries:
                if args.rerun_finished or not entry["finish_path"].exists():
                    print(entry["command"])
        return

    log_dir = REPO_ROOT / "logs" / "cherimoya_adapter_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_finished_jobs = 0
    skipped_finished_commands = 0
    for (pseudobulk, fold), entries in commands_by_job.items():
        commands = [
            entry["command"]
            for entry in entries
            if args.rerun_finished or not entry["finish_path"].exists()
        ]
        skipped_finished_commands += len(entries) - len(commands)
        if not commands:
            skipped_finished_jobs += 1
            continue

        job_name = f"adapter_sweep_{args.dataset}_{args.sweep_prefix}_{pseudobulk}_f{fold}"
        command_text = "\n".join(commands)
        if args.scheduler_gpus is None:
            run_block = "\n".join(f"time {cmd}" for cmd in commands)
        else:
            gpu_args = " ".join(shlex.quote(gpu) for gpu in args.scheduler_gpus)
            run_block = (
                f"cat <<'COMMANDS' | simple_gpu_scheduler --gpus {gpu_args}\n"
                f"{command_text}\n"
                "COMMANDS"
            )

        sbatch_script = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus={args.gpus}
#SBATCH -C {args.constraint}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err

source ~/bs.sh
mamba activate plant-design
export NUMBA_NUM_THREADS={args.numba_threads}
nvidia-smi -L

{run_block}
"""

        if args.dry_run:
            print(f"--- {job_name} ({len(commands)} commands) ---")
            print(sbatch_script)
            submitted += 1
            continue

        result = subprocess.run(
            ["sbatch"], input=sbatch_script, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"ERROR submitting {job_name}: {result.stderr}", file=sys.stderr)
            continue
        submitted += 1
        print(f"{job_name}: {result.stdout.strip()}")

    if args.dry_run:
        print(f"\nDry run: generated {submitted} sbatch job(s).")
    else:
        print(f"\nSubmitted {submitted} sbatch job(s).")
    if skipped_finished_commands:
        print(
            f"Skipped {skipped_finished_commands} finished command(s) "
            f"across {skipped_finished_jobs} completed job group(s)."
        )


if __name__ == "__main__":
    main()
