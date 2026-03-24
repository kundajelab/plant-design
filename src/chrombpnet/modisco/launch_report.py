#!/usr/bin/env python3
"""Submit SLURM jobs to run modisco report on ChromBPNet modisco h5 outputs.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, model_type, head) triple via `modisco report`.

Jobs are skipped if the report output already exists or if the required modisco
h5 file is missing (run launch.py first).

Usage:
    python src/chrombpnet/modisco/launch_report.py                          # all pseudobulks, both model types, both heads
    python src/chrombpnet/modisco/launch_report.py --dry-run                # print sbatch scripts without submitting
    python src/chrombpnet/modisco/launch_report.py --head count             # count head only
    python src/chrombpnet/modisco/launch_report.py --model-type bias        # bias model only
    python src/chrombpnet/modisco/launch_report.py --time 2:00:00 --mem 32G
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS_PATH = REPO_ROOT / "configs" / "pseudobulks.txt"
JASPAR_PATH = REPO_ROOT / "data" / "JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--model-type",
        type=str,
        action="append",
        choices=["bias", "accessibility"],
        default=None,
        metavar="MODEL_TYPE",
        help="model type(s) to run; repeatable (default: bias accessibility)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="4:00:00")
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]
    model_types = (
        args.model_type if args.model_type is not None else ["bias", "accessibility"]
    )

    # Load pseudobulk list
    pseudobulks = PSEUDOBULKS_PATH.read_text().splitlines()
    pseudobulks = [p for p in pseudobulks if p.strip()]

    out_dir = REPO_ROOT / "modisco" / "chrombpnet"
    log_dir = REPO_ROOT / "logs" / "chrombpnet_modisco_report"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_h5 = 0

    for pseudobulk in pseudobulks:
        for model_type in model_types:
            for head in heads:
                out_h5 = out_dir / f"{pseudobulk}_{model_type}_{head}.modisco.h5"
                if not out_h5.exists():
                    skipped_no_h5 += 1
                    continue

                report_dir = out_dir / f"{pseudobulk}_{model_type}_{head}.modisco"
                if report_dir.exists():
                    skipped_done += 1
                    continue

                job_name = f"modisco_report_{pseudobulk}_{model_type}_{head}"

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
    total = len(pseudobulks) * len(model_types) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_no_h5} missing h5 files, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
