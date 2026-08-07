#!/usr/bin/env python3
"""Development-only launcher for ChromBPNet low-rank adapter sweep benchmarks.

The launcher checks for the expected benchmark artifacts and only submits jobs
for missing outputs unless --rerun is used. Jobs are grouped by model/sweep ID
so a large sweep creates one SLURM job per LoRA sweep plus up to three reference
jobs, rather than one job per output file.

Use launch.py for the active benchmark path.

Plot finished outputs with:
    uv run python src/chrombpnet/benchmark/plot_lora_sweep.py --sweep-id <sweep_id>
"""

import argparse
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "src" / "cherimoya" / "benchmark"
PYTHON = ("uv", "run", "python")
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
REFERENCE_MODELS = {
    "base": {
        "label": "Base",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted.json",
        "flags": [],
    },
    "finetuned": {
        "label": "Finetuned 50:50",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark_finetuned.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned.json",
        "flags": ["--finetuned"],
    },
    "finetuned_0.8": {
        "label": "Finetuned 80:20",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned_0.8",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark_finetuned_0.8.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned_0.8.json",
        "flags": ["--finetuned", "--da-fraction", "0.8"],
    },
}


def discover_sweep_ids():
    sweep_ids = set()
    for path in (REPO_ROOT / "models").glob("*/adapters/*"):
        if path.is_dir():
            sweep_ids.add(path.name)
    for path in (REPO_ROOT / "performance_metrics" / "adapters").glob("*"):
        if path.is_dir():
            sweep_ids.add(path.name)
    return sorted(sweep_ids)


def slurm_job_name(prefix, name, max_len=120):
    raw = f"{prefix}_{name}"
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    keep = max_len - len(prefix) - len(digest) - 2
    return f"{prefix}_{name[:keep]}_{digest}"


def existing_cross_path(spec, pseudobulk):
    out_path = spec["cross_dir"] / f"{pseudobulk}.json"
    return out_path if out_path.exists() else None


def benchmark_da_peaks_path(da_path):
    return da_path.with_name(
        da_path.stem.replace("da_benchmark", "da_benchmark_peaks") + da_path.suffix
    )


def reference_job(args, model_id, spec):
    cross = []
    serial = []
    skipped = 0

    for pseudobulk in args.pseudobulks:
        if existing_cross_path(spec, pseudobulk) is not None and not args.rerun:
            skipped += 1
            continue
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark.py",
            "-p",
            pseudobulk,
            "-b",
            args.batch_size,
            *spec["flags"],
        ]
        if args.save_cross_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        cross.append(" ".join(shlex.quote(str(part)) for part in cmd))

    da_path = spec["da_path"]
    da_done = da_path.exists() and (
        not args.save_da_outputs or benchmark_da_peaks_path(da_path).exists()
    )
    if not da_done or args.rerun:
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark_da.py",
            "-b",
            args.batch_size,
            *spec["flags"],
        ]
        if args.save_da_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        serial.append(" ".join(shlex.quote(str(part)) for part in cmd))
    else:
        skipped += 1

    if not spec["pairwise_path"].exists() or args.rerun:
        serial.append(
            " ".join(
                shlex.quote(str(part))
                for part in [
                    *PYTHON,
                    BENCHMARK_DIR / "pairwise_correlations_predicted.py",
                    *spec["flags"],
                ]
            )
        )
    else:
        skipped += 1

    return {
        "job_name": slurm_job_name("benchmark_lora_ref", model_id),
        "cross": cross,
        "serial": serial,
        "skipped": skipped,
    }


def adapter_job(args, sweep_id):
    cross = []
    serial = []
    skipped = 0

    for pseudobulk in args.pseudobulks:
        out_path = REPO_ROOT / "performance_metrics" / "adapters" / sweep_id / f"{pseudobulk}.json"
        if out_path.exists() and not args.rerun:
            skipped += 1
            continue
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark.py",
            "-p",
            pseudobulk,
            "--adapter-sweep-id",
            sweep_id,
            "-b",
            args.batch_size,
        ]
        if args.save_cross_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        cross.append(" ".join(shlex.quote(str(part)) for part in cmd))

    da_json = (
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / f"da_benchmark_adapter_{sweep_id}.json"
    )
    da_peaks = (
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / f"da_benchmark_peaks_adapter_{sweep_id}.csv"
    )
    da_done = da_json.exists() and (not args.save_da_outputs or da_peaks.exists())
    if not da_done or args.rerun:
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark_da.py",
            "--adapter-sweep-id",
            sweep_id,
            "-b",
            args.batch_size,
        ]
        if args.save_da_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        serial.append(" ".join(shlex.quote(str(part)) for part in cmd))
    else:
        skipped += 1

    pairwise_json = (
        REPO_ROOT
        / "performance_metrics"
        / "adapters"
        / sweep_id
        / f"pairwise_correlations_predicted_adapter_{sweep_id}.json"
    )
    if not pairwise_json.exists() or args.rerun:
        serial.append(
            " ".join(
                shlex.quote(str(part))
                for part in [
                    *PYTHON,
                    BENCHMARK_DIR / "pairwise_correlations_predicted.py",
                    "--adapter-sweep-id",
                    sweep_id,
                ]
            )
        )
    else:
        skipped += 1

    return {
        "job_name": slurm_job_name("benchmark_lora", sweep_id),
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
mamba activate plant-design
export NUMBA_NUM_THREADS={args.numba_threads}
nvidia-smi -L

{run_block}
"""


def submit_jobs(args, jobs):
    log_dir = REPO_ROOT / "logs" / "chrombpnet_lora_benchmark"
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
    parser.add_argument("--sweep-id", action="append", default=None)
    parser.add_argument(
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        choices=PSEUDOBULKS,
        default=None,
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Print missing uv run benchmark commands without sbatch wrappers.",
    )
    parser.add_argument("--save-cross-outputs", action="store_true")
    parser.add_argument(
        "--no-save-da-outputs",
        dest="save_da_outputs",
        action="store_false",
        help="Do not pass --save-output to benchmark_da.py.",
    )
    parser.set_defaults(save_da_outputs=True)
    parser.add_argument("--no-references", dest="include_references", action="store_false")
    parser.set_defaults(include_references=True)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--partition", type=str, default="akundaje,gpu,owners")
    parser.add_argument("--constraint", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument("--numba-threads", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    args.pseudobulks = args.pseudobulks or PSEUDOBULKS
    sweep_ids = args.sweep_id or discover_sweep_ids()
    if not sweep_ids and not args.include_references:
        print(
            "Error: no adapter sweep IDs provided or discovered, and references are disabled.",
            file=sys.stderr,
        )
        sys.exit(1)

    jobs = []
    if args.include_references:
        jobs.extend(reference_job(args, model_id, spec) for model_id, spec in REFERENCE_MODELS.items())
    jobs.extend(adapter_job(args, sweep_id) for sweep_id in sweep_ids)
    submit_jobs(args, jobs)


if __name__ == "__main__":
    main()
