#!/usr/bin/env python3
"""Launch SLURM benchmark jobs for active ChromBPNet model variants.

This launcher covers the main reporting path:

- base ChromBPNet
- full-model DA fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter DA fine-tuning at 0.5 and 0.8 DA fractions

Development-only adapter parameter sweeps are handled separately by
``benchmark_lora_sweep.py``.
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "src" / "chrombpnet" / "benchmark"
DATASETS = ("base", "fine", "coarse")


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    flags: tuple[str, ...]
    cross_dir: Path
    da_path: Path
    pairwise_path: Path


def adapter_variant(name, label, run_id, da_fraction):
    return Variant(
        name=name,
        label=label,
        flags=("--adapter", "--da-fraction", str(da_fraction)),
        cross_dir=REPO_ROOT / "performance_metrics" / "adapters" / run_id,
        da_path=REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / f"da_benchmark_adapter_{run_id}.json",
        pairwise_path=REPO_ROOT
        / "performance_metrics"
        / "adapters"
        / run_id
        / f"pairwise_correlations_predicted_adapter_{run_id}.json",
    )


VARIANTS = {
    "base": Variant(
        name="base",
        label="Base",
        flags=(),
        cross_dir=REPO_ROOT / "performance_metrics" / "chrombpnet",
        da_path=REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark.json",
        pairwise_path=REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted.json",
    ),
    "finetuned": Variant(
        name="finetuned",
        label="Finetuned 50:50",
        flags=("--finetuned",),
        cross_dir=REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned",
        da_path=REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark_finetuned.json",
        pairwise_path=REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned.json",
    ),
    "finetuned_0.8": Variant(
        name="finetuned_0.8",
        label="Finetuned 80:20",
        flags=("--finetuned", "--da-fraction", "0.8"),
        cross_dir=REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned_0.8",
        da_path=REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark_finetuned_0.8.json",
        pairwise_path=REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned_0.8.json",
    ),
    "adapter": adapter_variant(
        "adapter",
        "Adapter 50:50",
        "daicon_default_da0p5",
        0.5,
    ),
    "adapter_0.8": adapter_variant(
        "adapter_0.8",
        "Adapter 80:20",
        "daicon_default_da0p8",
        0.8,
    ),
}


def slurm_name(text):
    return (
        text.replace("/", "_")
        .replace(".", "p")
        .replace("+", "_")
        .replace("-", "_")
    )


def read_pseudobulks(dataset):
    suffix = "" if dataset == "base" else f"_{dataset}"
    path = REPO_ROOT / "configs" / f"pseudobulks{suffix}.txt"
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def existing_cross_path(dataset_name, variant, pseudobulk):
    metrics_subdir = variant.cross_dir.relative_to(REPO_ROOT / "performance_metrics")
    path = (
        REPO_ROOT
        / "performance_metrics"
        / (
            metrics_subdir
            if dataset_name == "base"
            else f"{metrics_subdir}_{dataset_name}"
        )
        / f"{pseudobulk}.json"
    )
    return path if path.exists() else None


def benchmark_da_peaks_path(da_path):
    return da_path.with_name(
        da_path.stem.replace("da_benchmark", "da_benchmark_peaks") + da_path.suffix
    )


def build_variant_job(args, variant):
    cross = []
    serial = []
    skipped = 0

    for pseudobulk in args.pseudobulks:
        if existing_cross_path(args.dataset, variant, pseudobulk) is not None and not args.rerun:
            skipped += 1
            continue
        cmd = [
            "python",
            BENCHMARK_DIR / "benchmark.py",
            "--dataset",
            args.dataset,
            "-p",
            pseudobulk,
            "-b",
            args.batch_size,
            *variant.flags,
        ]
        if args.save_cross_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        cross.append(" ".join(shlex.quote(str(part)) for part in cmd))

    if args.dataset != "base":
        return {
            "job_name": slurm_name(f"benchmark_{args.dataset}_{variant.name}"),
            "cross": cross,
            "serial": serial,
            "skipped": skipped,
        }

    da_done = variant.da_path.exists() and (
        not args.save_da_outputs or benchmark_da_peaks_path(variant.da_path).exists()
    )
    if not da_done or args.rerun:
        cmd = [
            "python",
            BENCHMARK_DIR / "benchmark_da.py",
            "-b",
            args.batch_size,
            *variant.flags,
        ]
        if args.save_da_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        serial.append(" ".join(shlex.quote(str(part)) for part in cmd))
    else:
        skipped += 1

    if not variant.pairwise_path.exists() or args.rerun:
        serial.append(
            " ".join(
                shlex.quote(str(part))
                for part in [
                    "python",
                    BENCHMARK_DIR / "pairwise_correlations_predicted.py",
                    *variant.flags,
                ]
            )
        )
    else:
        skipped += 1

    return {
        "job_name": slurm_name(f"benchmark_{args.dataset}_{variant.name}"),
        "cross": cross,
        "serial": serial,
        "skipped": skipped,
    }


def make_run_block(job):
    commands = job["cross"] + job["serial"]
    return "\n\n".join(f"time {cmd}" for cmd in commands)


def make_sbatch_script(args, job, log_dir):
    run_block = make_run_block(job)
    return f"""#!/bin/bash -l
#SBATCH --job-name={job["job_name"]}
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C {args.constraint}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --output={log_dir}/{job["job_name"]}.out
#SBATCH --error={log_dir}/{job["job_name"]}.err

source ~/bs.sh
mamba activate torch
export NUMBA_NUM_THREADS={args.numba_threads}
nvidia-smi -L

{run_block}
"""


def submit_jobs(args, jobs):
    log_dir = REPO_ROOT / "logs" / "chrombpnet_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_empty = 0
    for job in jobs:
        n_commands = len(job["cross"]) + len(job["serial"])
        if n_commands == 0:
            skipped_empty += 1
            continue

        script = make_sbatch_script(args, job, log_dir)
        if args.interactive:
            for command in job["cross"] + job["serial"]:
                print(command)
            continue
        if args.dry_run:
            print(f"--- {job['job_name']} ({n_commands} commands, skipped {job['skipped']}) ---")
            print(script)
            submitted += 1
            continue

        result = subprocess.run(["sbatch"], input=script, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR submitting {job['job_name']}: {result.stderr}", file=sys.stderr)
            continue
        submitted += 1
        print(f"{job['job_name']}: {result.stdout.strip()}")

    if args.interactive:
        return
    if args.dry_run:
        print(f"\nDry run: generated {submitted} sbatch job(s); skipped {skipped_empty} empty job group(s).")
    else:
        print(f"\nSubmitted {submitted} sbatch job(s); skipped {skipped_empty} empty job group(s).")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        default=None,
        help="Model variant to benchmark; repeatable. Default: all active variants.",
    )
    parser.add_argument(
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to benchmark (default: base).",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Print missing benchmark python commands without sbatch wrappers.",
    )
    parser.add_argument("--save-cross-outputs", action="store_true")
    parser.add_argument(
        "--no-save-da-outputs",
        dest="save_da_outputs",
        action="store_false",
        help="Do not pass --save-output to benchmark_da.py.",
    )
    parser.set_defaults(save_da_outputs=True)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--partition", type=str, default="akundaje,gpu,owners")
    parser.add_argument("--constraint", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument("--numba-threads", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    choices = read_pseudobulks(args.dataset)
    args.pseudobulks = args.pseudobulks or choices
    for pseudobulk in args.pseudobulks:
        if pseudobulk not in choices:
            print(
                f"Error: unknown pseudobulk '{pseudobulk}' for dataset '{args.dataset}'. "
                f"Choose one of: {', '.join(choices)}",
                file=sys.stderr,
            )
            sys.exit(1)
    variant_names = args.variant or (["base"] if args.dataset != "base" else list(VARIANTS))
    jobs = [build_variant_job(args, VARIANTS[name]) for name in variant_names]
    submit_jobs(args, jobs)


if __name__ == "__main__":
    main()
