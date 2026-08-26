#!/usr/bin/env python3
"""Submit SLURM jobs to generate Cherimoya GC-matched negative regions.

Negatives are sampled genome-wide per pseudobulk (not per fold): Cherimoya's
own `negatives` CLI has no chromosome-split concept, and PeakGenerator/
extract_loci already restrict extraction to the active fold's chromosomes at
train time, so one negatives BED per pseudobulk covers every fold. Each job
runs one pseudobulk; CPU-only (no GPU requested by default).

By default, jobs run via `apptainer exec` against the Cherimoya Apptainer
image (see ../apptainer/) -- no `--nv` since this doesn't need a GPU.

--local skips SLURM entirely and runs each pseudobulk directly in the
foreground with inherited stdout/stderr, instead of submitting SLURM jobs.
Combined with --dry-run, --local prints one runnable command per pseudobulk
instead of running them -- pipe that into something like `parallel`.

Usage:
    python src/cherimoya_pipeline/fit/negatives.py
    python src/cherimoya_pipeline/fit/negatives.py --dry-run
    python src/cherimoya_pipeline/fit/negatives.py --local
    python src/cherimoya_pipeline/fit/negatives.py --local --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap

from launcher_common import REPO_ROOT, add_common_args, resolve_dataset_and_jobs

import species_config


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each pseudobulk directly in the foreground, instead of "
            "submitting SLURM jobs; ignores SLURM resource flags "
            "(--partition/--cpus-per-task/--mem/--time); combined with "
            "--dry-run, prints one command per pseudobulk instead of "
            "running them"
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        default=default_apptainer_image(),
        help=f"path to the Cherimoya Apptainer .sif image (default: {default_apptainer_image()})",
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=None,
        help=(
            "path to bind into the Apptainer container; may be repeated "
            "(default: the repo root)"
        ),
    )
    # SLURM resource flags (CPU-only job, no --gpus)
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="16G")
    parser.add_argument("--time", type=str, default="4:00:00")
    return parser.parse_args()


def negatives_command_parts(species, dataset, pseudobulk, out_path):
    dataset_suffix = "" if dataset == "base" else f"_{dataset}"
    data_dir = species_config.data_dir(species)
    return [
        "cherimoya",
        "negatives",
        "-i",
        str(data_dir / f"peaks{dataset_suffix}" / f"{pseudobulk}_peaks.narrowPeak"),
        "-o",
        str(out_path),
        "-f",
        str(species_config.genome_fasta(species)),
        "-b",
        str(data_dir / f"bigwigs{dataset_suffix}" / f"{pseudobulk}.bw"),
        "-v",
    ]


def main():
    args = parse_args()
    species, pseudobulks, _folds = resolve_dataset_and_jobs(args)
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    negatives_dir = species_config.data_dir(species) / f"negatives{dataset_suffix}"
    negatives_dir.mkdir(parents=True, exist_ok=True)
    log_dir = species_config.logs_dir(species) / "cherimoya_negatives"
    log_dir.mkdir(parents=True, exist_ok=True)

    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)

    submitted = 0
    skipped = 0
    local_failures = []

    for pseudobulk in pseudobulks:
        out_path = negatives_dir / f"{pseudobulk}_negatives.bed"
        if not args.rerun and out_path.exists():
            skipped += 1
            continue

        command_parts = negatives_command_parts(species, args.dataset, pseudobulk, out_path)

        if args.local:
            cmd = [
                "apptainer",
                "exec",
                *[part for path in apptainer_binds for part in ("--bind", path)],
                args.apptainer_image,
                *command_parts,
            ]
            if args.dry_run:
                print(shlex.join(cmd))
                submitted += 1
                continue

            print(f"=== Generating negatives for {pseudobulk} ===")
            result = subprocess.run(cmd, cwd=REPO_ROOT)
            if result.returncode != 0:
                local_failures.append((pseudobulk, result.returncode))
                print(
                    f"ERROR: {pseudobulk} exited with {result.returncode}",
                    file=sys.stderr,
                )
            submitted += 1
            continue

        job_name = f"cherimoya_negatives_{args.dataset}_{pseudobulk}".replace(".", "p")
        command_bash = " ".join(shlex.quote(part) for part in command_parts)
        negatives_cmd = (
            f'apptainer exec {apptainer_bind_args} "$APPTAINER_IMAGE" {command_bash}'
        )

        sbatch_script = textwrap.dedent(f"""\
            #!/bin/bash -l
            #SBATCH --job-name={job_name}
            #SBATCH --ntasks=1
            #SBATCH --ntasks-per-node=1
            #SBATCH --nodes=1
            #SBATCH --cpus-per-task={args.cpus_per_task}
            #SBATCH --mem={args.mem}
            #SBATCH --partition={args.partition}
            #SBATCH --time={args.time}
            #SBATCH --output={log_dir}/{job_name}.out
            #SBATCH --error={log_dir}/{job_name}.err

            APPTAINER_IMAGE="{args.apptainer_image}"

            {negatives_cmd}
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
                f"ERROR submitting {job_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
    print(
        f"\n{action} {submitted} negatives job(s), skipped {skipped} "
        "already-complete pseudobulk(s)."
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} pseudobulk(s) failed:")
            for pseudobulk, returncode in local_failures:
                print(f"  {pseudobulk} (exit {returncode})")
        else:
            print("All pseudobulks completed successfully.")


if __name__ == "__main__":
    main()
