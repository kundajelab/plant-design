#!/usr/bin/env python3
"""Submit standard DA fine-tuning jobs.

This launcher is for the small set of named runs used downstream:

- full-model fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter fine-tuning at 0.5 and 0.8 DA fractions

Use ``launch_adapter_sweep.py`` for the full adapter hyperparameter sweep.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS_PATH = REPO_ROOT / "configs" / "pseudobulks.txt"
N_FOLDS = 5
DA_FRACTIONS = (0.5, 0.8)

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


def slurm_name(text):
    return (
        text.replace("/", "_")
        .replace(".", "p")
        .replace("+", "_")
        .replace("-", "_")
    )


def load_pseudobulks(selected):
    if selected is not None:
        return selected
    return [p for p in PSEUDOBULKS_PATH.read_text().splitlines() if p.strip()]


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


def build_full_job(args, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "chrombpnet" / "da_finetuning" / "finetune_da.py"
    command = [
        "python",
        script,
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
        REPO_ROOT
        / "models"
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


def build_adapter_job(args, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "chrombpnet" / "da_finetuning" / "finetune_da.py"
    run_id = default_adapter_run_id(da_fraction)
    command = [
        "python",
        script,
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
        REPO_ROOT
        / "models"
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


def build_jobs(args):
    pseudobulks = load_pseudobulks(args.pseudobulk)
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
                    builders[run_group](args, pseudobulk, fold, da_fraction)
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


def submit_jobs(args, jobs):
    log_dir = REPO_ROOT / "logs" / "chrombpnet_da_finetuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    submitted = 0
    skipped_finished_jobs = 0
    skipped_finished_commands = 0

    for (run_group, pseudobulk, fold), entries in jobs.items():
        commands = [
            entry["command"]
            for entry in entries
            if args.rerun_finished or not entry["finish_path"].exists()
        ]
        skipped_finished_commands += len(entries) - len(commands)
        if not commands:
            skipped_finished_jobs += 1
            continue

        job_name = slurm_name(f"da_ft_{run_group}_{pseudobulk}_f{fold}")
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
mamba activate torch
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        choices=RUN_GROUPS,
        default=None,
        help="Run group to submit; repeatable. Default: all groups.",
    )
    parser.add_argument("--pseudobulk", action="append", default=None)
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
        help="report expected training logs and exit",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="include commands whose expected training log already exists",
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

    if args.run is None:
        args.run = list(RUN_GROUPS)

    jobs = build_jobs(args)
    if args.check_finished:
        print_finished_report(jobs)
        return

    if args.interactive:
        for entries in jobs.values():
            for entry in entries:
                if args.rerun_finished or not entry["finish_path"].exists():
                    print(entry["command"])
        return

    submit_jobs(args, jobs)


if __name__ == "__main__":
    main()
