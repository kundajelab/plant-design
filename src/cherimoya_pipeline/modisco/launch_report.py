#!/usr/bin/env python3
"""Submit SLURM jobs to run modisco report on Cherimoya modisco h5 outputs.

Reads pseudobulks from configs/<species>/pseudobulks.txt and submits one
sbatch job per (pseudobulk, head) pair via `modisco report`.

Jobs are skipped if the report output already exists or if the required modisco
h5 file is missing (run launch.py first).

`modisco` (tf-modisco-lite) is pip-installed directly inside the Cherimoya
Apptainer image (see ../apptainer/cherimoya.def) -- by default, jobs run
`modisco report` via `apptainer exec` against that image (no GPU needed).

--local skips SLURM entirely and runs each (pseudobulk, head) job directly
in the foreground with inherited stdout/stderr, instead of submitting SLURM
jobs. Combined with --dry-run, --local prints one runnable command per job
instead of running them.

Usage:
    python src/cherimoya_pipeline/modisco/launch_report.py                                    # all pseudobulks, both heads
    python src/cherimoya_pipeline/modisco/launch_report.py --dry-run                          # print sbatch scripts without submitting
    python src/cherimoya_pipeline/modisco/launch_report.py --head count                       # count head only
    python src/cherimoya_pipeline/modisco/launch_report.py --finetuned                        # reports for finetuned model attributions (50:50)
    python src/cherimoya_pipeline/modisco/launch_report.py --finetuned --da-fraction 0.8     # reports for finetuned model attributions (80:20)
    python src/cherimoya_pipeline/modisco/launch_report.py --adapter                         # reports for default daicon adapter attributions (50:50)
    python src/cherimoya_pipeline/modisco/launch_report.py --adapter --da-fraction 0.8       # reports for default daicon adapter attributions (80:20)
    python src/cherimoya_pipeline/modisco/launch_report.py --da-peaks tau                    # reports for tau DA-peaks attributions
    python src/cherimoya_pipeline/modisco/launch_report.py --da-peaks specific_union       # reports for per-cell-type union DA-peaks attributions
    python src/cherimoya_pipeline/modisco/launch_report.py --time 2:00:00 --mem 32G
    python src/cherimoya_pipeline/modisco/launch_report.py --local --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attribute"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from da_peaks import DA_PEAK_CHOICES, da_suffix, read_pseudobulks
from model_variants import add_model_variant_args, resolve_model_variant

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

JASPAR_PATH = REPO_ROOT / "data" / "JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt"


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
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
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution head(s) to run; repeatable (default: profile count)",
    )
    add_model_variant_args(
        parser,
        finetuned_help="Use modisco outputs from finetuned model attributions instead of base models",
        adapter_help="Use modisco outputs from default daicon adapter attributions for the selected DA fraction",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run reports on DA-peaks-only modisco outputs (default: all peaks)",
    )
    # SLURM resource flags (CPU-only job)
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="4:00:00")
    args = parser.parse_args()
    species = species_config.resolve_species(args.species)
    variant = resolve_model_variant(args)
    fine_tag = variant["fine_tag"]
    adapter_id = variant["adapter_id"]

    heads = args.head if args.head is not None else ["profile", "count"]

    if adapter_id is not None:
        modisco_subdir = f"cherimoya_adapter_{adapter_id}"
    else:
        modisco_subdir = f"cherimoya_{fine_tag}" if args.finetuned else "cherimoya"
    suffix = da_suffix(args.da_peaks)

    pseudobulks = read_pseudobulks(species, "base")

    out_dir = species_config.modisco_dir(species) / modisco_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = species_config.logs_dir(species) / "cherimoya_modisco_report"
    log_dir.mkdir(parents=True, exist_ok=True)
    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)

    submitted = 0
    skipped_done = 0
    skipped_no_h5 = 0
    local_failures = []

    for pseudobulk in pseudobulks:
        for head in heads:
            out_h5 = out_dir / f"{pseudobulk}_{head}{suffix}.modisco.h5"
            if not out_h5.exists():
                skipped_no_h5 += 1
                continue

            # Absolute path: modisco report -o interprets a relative path
            # relative to the process's cwd, not -i's directory, so a
            # relative report_dir would land wherever the launcher happens
            # to be invoked from instead of alongside the h5 file.
            report_dir = out_dir / f"{pseudobulk}_{head}{suffix}.modisco"
            if report_dir.exists():
                skipped_done += 1
                continue

            if adapter_id is not None:
                tag = f"_adapter_{adapter_id}"
            else:
                tag = f"_{fine_tag}" if args.finetuned else ""
            job_name = f"modisco_report{tag}{suffix}_{pseudobulk}_{head}".replace(".", "p")

            command = [
                "modisco",
                "report",
                "-l",
                "-i",
                str(out_h5),
                "-o",
                str(report_dir),
                "-m",
                str(JASPAR_PATH),
            ]

            if args.local:
                full_cmd = [
                    "apptainer",
                    "exec",
                    *[part for path in apptainer_binds for part in ("--bind", path)],
                    args.apptainer_image,
                    *command,
                ]
                if args.dry_run:
                    print(shlex.join(full_cmd))
                    submitted += 1
                    continue
                print(f"=== {job_name} ===")
                result = subprocess.run(full_cmd, cwd=REPO_ROOT)
                if result.returncode != 0:
                    local_failures.append((job_name, result.returncode))
                    print(f"ERROR: {job_name} exited with {result.returncode}", file=sys.stderr)
                submitted += 1
                continue

            command_bash = " ".join(shlex.quote(part) for part in command)
            modisco_report_cmd = (
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
                #SBATCH -C NO_GPU

                APPTAINER_IMAGE="{args.apptainer_image}"

                time {modisco_report_cmd}
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
    total = len(pseudobulks) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_no_h5} missing h5 files, "
        f"skipped {skipped_done} already done ({total} total)"
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} job(s) failed:")
            for job_name, returncode in local_failures:
                print(f"  {job_name} (exit {returncode})")
        else:
            print("All jobs completed successfully.")


if __name__ == "__main__":
    main()
