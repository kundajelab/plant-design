#!/usr/bin/env python3
"""Submit SLURM jobs to run modisco report on ChromBPNet modisco h5 outputs.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, model_type, head) triple via `modisco report`.

Jobs are skipped if the report output already exists or if the required modisco
h5 file is missing (run launch.py first).

Usage:
    python src/chrombpnet/modisco/launch_report.py                                    # all pseudobulks, both model types, both heads
    python src/chrombpnet/modisco/launch_report.py --dry-run                          # print sbatch scripts without submitting
    python src/chrombpnet/modisco/launch_report.py --head count                       # count head only
    python src/chrombpnet/modisco/launch_report.py --model-type bias                  # bias model only
    python src/chrombpnet/modisco/launch_report.py --finetuned                        # reports for finetuned model attributions (50:50)
    python src/chrombpnet/modisco/launch_report.py --finetuned --da-fraction 0.8     # reports for finetuned model attributions (80:20)
    python src/chrombpnet/modisco/launch_report.py --da-peaks tau                    # reports for tau DA-peaks attributions
    python src/chrombpnet/modisco/launch_report.py --da-peaks specific_union       # reports for per-cell-type union DA-peaks attributions
    python src/chrombpnet/modisco/launch_report.py --time 2:00:00 --mem 32G
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attribute"))
from da_peaks import DA_PEAK_CHOICES, da_suffix

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS_PATH = REPO_ROOT / "configs" / "pseudobulks.txt"
JASPAR_PATH = REPO_ROOT / "data" / "JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
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
        "--model",
        type=str,
        action="append",
        choices=["bias", "accessibility"],
        default=None,
        metavar="MODEL_TYPE",
        help="model type(s) to run; repeatable (default: bias accessibility)",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use modisco outputs from finetuned model attributions instead of base models",
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects modisco subdirectory (default: 0.5)",
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

    heads = args.head if args.head is not None else ["profile", "count"]
    model_types = (
        args.model_type if args.model_type is not None else ["bias", "accessibility"]
    )

    fine_tag = "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    modisco_subdir = f"chrombpnet_{fine_tag}" if args.finetuned else "chrombpnet"
    suffix = da_suffix(args.da_peaks)

    if args.finetuned and "bias" in model_types:
        print(
            "Warning: bias model is not saved during finetuning; skipping bias model type.",
            file=sys.stderr,
        )
        model_types = [m for m in model_types if m != "bias"]
        if not model_types:
            sys.exit(0)

    # Load pseudobulk list
    pseudobulks = PSEUDOBULKS_PATH.read_text().splitlines()
    pseudobulks = [p for p in pseudobulks if p.strip()]

    out_dir = REPO_ROOT / "modisco" / modisco_subdir
    log_dir = REPO_ROOT / "logs" / "chrombpnet_modisco_report"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_h5 = 0

    for pseudobulk in pseudobulks:
        for model_type in model_types:
            for head in heads:
                out_h5 = out_dir / f"{pseudobulk}_{model_type}_{head}{suffix}.modisco.h5"
                if not out_h5.exists():
                    skipped_no_h5 += 1
                    continue

                report_dir = out_dir / f"{pseudobulk}_{model_type}_{head}{suffix}.modisco"
                if report_dir.exists():
                    skipped_done += 1
                    continue

                tag = f"_{fine_tag}" if args.finetuned else ""
                job_name = f"modisco_report{tag}{suffix}_{pseudobulk}_{model_type}_{head}"

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
