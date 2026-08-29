#!/usr/bin/env python3
"""Submit one SLURM adapter sweep job per pseudobulk, fold, and hyperparameter
combo. For development/optimization of adapter hyperparameters.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each job directly in the foreground
with inherited stdout/stderr, instead of submitting SLURM jobs. Combined
with --dry-run, --local prints one runnable command per job instead of
running them -- pipe that into something like `simple_gpu_scheduler` to fan
a sweep out across a multi-GPU box.
"""

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASETS = ("base", "fine", "coarse")
N_FOLDS = 5


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


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def model_root(species, dataset):
    root = species_config.models_dir(species)
    return root if dataset == "base" else root / dataset


def build_jobs(args, species, sweep):
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

    script = (
        REPO_ROOT
        / "src"
        / "cherimoya_pipeline"
        / "da_finetuning"
        / "finetune_adapters.py"
    )
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

    jobs = []
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
                job_name = f"da_sweep_{sweep_id}_{pseudobulk}_f{fold}".replace(".", "p")
                jobs.append(
                    {
                        "job_name": job_name,
                        "command": cmd,
                        "finish_path": log_path,
                        "sweep_id": sweep_id,
                        "pseudobulk": pseudobulk,
                        "fold": fold,
                    }
                )
    return jobs


def print_finished_report(jobs):
    total_finished = sum(job["finish_path"].exists() for job in jobs)
    for job in jobs:
        status = "DONE" if job["finish_path"].exists() else "PENDING"
        print(f"{status}\t{job['sweep_id']}\t{job['pseudobulk']}\tfold={job['fold']}")
        if status == "PENDING":
            print(f"MISSING\t{job['finish_path']}")
    print(f"\nFinished {total_finished}/{len(jobs)} expected training logs.")


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
        help="report expected adapter bundles and exit without submitting jobs",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="include jobs whose expected adapter bundle already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each job directly in the foreground, instead of "
            "submitting SLURM jobs; ignores SLURM resource flags"
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        default=default_apptainer_image(),
        help=f"path to the Cherimoya Apptainer .sif image (default: {default_apptainer_image()})",
    )
    parser.add_argument("--apptainer-bind", action="append", default=None)
    parser.add_argument(
        "--gpus",
        type=str,
        default="|".join(
            [
                "GPU_SKU:A100_PCIE",
                "GPU_SKU:A100_SXM4",
                "GPU_SKU:A40",
                "GPU_SKU:H100_SXM5",
                "GPU_SKU:H200_SXM5",
                "GPU_SKU:L40S",
                "GPU_SKU:RTX_3090",
            ]
        ),
    )
    parser.add_argument("--partition", type=str, default="gpu,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2-00:00:00")
    args = parser.parse_args()

    species = species_config.resolve_species(args.species)
    sweep = _load_sweep(args.config)
    jobs = build_jobs(args, species, sweep)

    if args.check_finished:
        print_finished_report(jobs)
        return

    log_dir = species_config.logs_dir(species) / "cherimoya_da_finetuning_sweep"
    log_dir.mkdir(parents=True, exist_ok=True)
    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)

    submitted = 0
    skipped = 0
    local_failures = []

    for job in jobs:
        if not args.rerun_finished and job["finish_path"].exists():
            skipped += 1
            continue

        job_name = job["job_name"]
        cmd = job["command"]

        if args.local:
            full_cmd = [
                "apptainer",
                "exec",
                "--nv",
                *[part for path in apptainer_binds for part in ("--bind", path)],
                args.apptainer_image,
                *cmd,
            ]
            if args.dry_run:
                print(shlex.join(full_cmd))
                submitted += 1
                continue
            print(f"=== {job_name} ===")
            result = subprocess.run(full_cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                local_failures.append((job_name, result.returncode))
                print(
                    f"ERROR: {job_name} exited with {result.returncode}",
                    file=sys.stderr,
                )
            submitted += 1
            continue

        command_bash = " ".join(shlex.quote(part) for part in cmd)
        run_cmd = f'apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" {command_bash}'

        sbatch_script = textwrap.dedent(f"""\
            #!/bin/bash -l
            #SBATCH --job-name={job_name}
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodes=1
            #SBATCH --gpus=1
            #SBATCH -C {args.gpus}
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem}
            #SBATCH --partition={args.partition}
            #SBATCH --time={args.time}
            #SBATCH --output={log_dir}/{job_name}.out
            #SBATCH --error={log_dir}/{job_name}.err

            APPTAINER_IMAGE="{args.apptainer_image}"

            nvidia-smi -L
            time {run_cmd}
        """)

        if args.dry_run:
            print(f"--- {job_name} ---")
            print(sbatch_script)
            submitted += 1
            continue

        result = subprocess.run(
            ["sbatch"], input=sbatch_script, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"{job_name}: {result.stdout.strip()}")
            submitted += 1
        else:
            print(
                f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr
            )

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
    print(f"\n{action} {submitted} job(s); skipped {skipped} already-finished job(s).")
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} job(s) failed:")
            for job_name, returncode in local_failures:
                print(f"  {job_name} (exit {returncode})")
        else:
            print("All jobs completed successfully.")


if __name__ == "__main__":
    main()
