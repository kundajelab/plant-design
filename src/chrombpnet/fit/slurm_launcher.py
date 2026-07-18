"""Shared helpers for ChromBPNet SLURM training launchers."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chrombpnet.datasets import DATASET_NAMES, get_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_mem: str,
    default_time: str,
) -> None:
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
        help="Submit jobs even when completion marker files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch scripts instead of submitting.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Print one local command per job instead of submitting SLURM jobs.",
    )
    parser.add_argument("--partition", type=str, default="akundaje,gpu,owners")
    parser.add_argument(
        "--constraint", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR"
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default=default_mem)
    parser.add_argument("--time", type=str, default=default_time)
    parser.add_argument("--numba-threads", type=int, default=4)
    parser.add_argument(
        "--env",
        dest="chrombpnet_env",
        type=str,
        default="chrombpnet",
        help="Conda/mamba environment for TensorFlow ChromBPNet training.",
    )


def resolve_dataset_and_jobs(args: argparse.Namespace):
    dataset = get_dataset(REPO_ROOT, args.dataset)
    pseudobulks = args.pseudobulks or dataset.pseudobulks()
    folds = args.folds or list(range(5))
    for pseudobulk in pseudobulks:
        try:
            dataset.validate_pseudobulk(pseudobulk)
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc
    return dataset, pseudobulks, folds


def sanitize_job_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name.replace(".", "p"))


def quote_cmd(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def wrap_command(parts: list[str | Path]) -> str:
    quoted = [shlex.quote(str(part)) for part in parts]
    option_start = next(
        (i for i, part in enumerate(quoted) if part.startswith("-")),
        len(quoted),
    )
    head, tail = quoted[:option_start], quoted[option_start:]
    if not tail:
        return "time " + " ".join(head)

    lines = ["time " + " ".join(head) + " \\"]
    for i in range(0, len(tail), 2):
        chunk = tail[i : i + 2]
        suffix = " \\" if i + 2 < len(tail) else ""
        lines.append("    " + " ".join(chunk) + suffix)
    return "\n".join(lines)


def make_sbatch_script(
    *,
    args: argparse.Namespace,
    job_name: str,
    log_dir: Path,
    command: list[str | Path],
) -> str:
    return f"""#!/bin/bash -l
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

set -euo pipefail

if [[ -f "${{HOME}}/bs.sh" ]]; then
    source "${{HOME}}/bs.sh"
fi
mamba activate {shlex.quote(args.chrombpnet_env)}
export NUMBA_NUM_THREADS={args.numba_threads}
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}:${{CONDA_PREFIX}}/lib"
nvidia-smi -L

cd {shlex.quote(str(SCRIPT_DIR))}
{wrap_command(command)}
"""


def submit_jobs(
    *,
    args: argparse.Namespace,
    log_dir: Path,
    jobs: list[tuple[str, list[str | Path]]],
    label: str,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    submitted = 0
    for job_name, command in jobs:
        if args.interactive:
            print(quote_cmd(command))
            submitted += 1
            continue

        sbatch_script = make_sbatch_script(
            args=args, job_name=job_name, log_dir=log_dir, command=command
        )
        if args.dry_run:
            print(f"--- {job_name} ---")
            print(sbatch_script)
            submitted += 1
            continue

        result = subprocess.run(
            ["sbatch"], input=sbatch_script, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(
                f"ERROR submitting {job_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            continue
        print(f"{job_name}: {result.stdout.strip()}")
        submitted += 1

    if args.interactive:
        action = "Printed"
    else:
        action = "Dry run: generated" if args.dry_run else "Submitted"
    print(f"{action} {submitted} {label} command(s).", file=sys.stderr)
