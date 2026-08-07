#!/usr/bin/env python3
"""Submit SLURM jobs to run modisco report on Cherimoya modisco h5 outputs.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, head) pair via `modisco report`.

Jobs are skipped if the report output already exists or if the required modisco
h5 file is missing (run launch.py first).

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
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
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
    # SLURM resource flags
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

    # Load pseudobulk list
    pseudobulks = read_pseudobulks(species, "base")

    out_dir = species_config.modisco_dir(species) / modisco_subdir
    log_dir = species_config.logs_dir(species) / "cherimoya_modisco_report"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_h5 = 0

    for pseudobulk in pseudobulks:
        for head in heads:
            out_h5 = out_dir / f"{pseudobulk}_{head}{suffix}.modisco.h5"
            if not out_h5.exists():
                skipped_no_h5 += 1
                continue

            report_dir = out_dir / f"{pseudobulk}_{head}{suffix}.modisco"
            if report_dir.exists():
                skipped_done += 1
                continue

            if adapter_id is not None:
                tag = f"_adapter_{adapter_id}"
            else:
                tag = f"_{fine_tag}" if args.finetuned else ""
            job_name = f"modisco_report{tag}{suffix}_{pseudobulk}_{head}"

            modisco_report_cmd = (
                f"modisco report -l"
                f" -i {out_h5}"
                f" -o {report_dir.name}"
                f" -m {JASPAR_PATH}"
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

                ml openblas/0.3.28
                ml xsimd/8.1.0
                ml xz/5.8.1
                ml hdf5/1.14.4
                ml arrow/22.0.0
                ml load py-pyarrow/18.1.0_py312
                ml lz4/1.8.0
                ml biology
                ml htslib
                ml ucsc-utils

                mamba activate torch

                export NUMBA_NUM_THREADS={args.cpus_per_task}
                mkdir -p {out_dir}
                cd {out_dir}
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

    action = "Would submit" if args.dry_run else "Submitted"
    total = len(pseudobulks) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_no_h5} missing h5 files, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
