#!/usr/bin/env python3
"""Submit SLURM jobs to run tf-modisco motifs on ChromBPNet attributions.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, model_type, head) triple via `modisco motifs`.

Jobs are skipped if the modisco h5 output already exists or if the required
attribution/OHE npz files are missing (run attribute/launch.py first).
Run launch_report.py afterwards to generate HTML reports from the h5 files.

Usage:
    python src/chrombpnet/modisco/launch.py                                    # all pseudobulks, both model types, both heads
    python src/chrombpnet/modisco/launch.py --dry-run                          # print sbatch scripts without submitting
    python src/chrombpnet/modisco/launch.py --head count                       # count head only
    python src/chrombpnet/modisco/launch.py --model-type bias                  # bias model only
    python src/chrombpnet/modisco/launch.py --finetuned                        # finetuned model attributions (50:50 DA split)
    python src/chrombpnet/modisco/launch.py --finetuned --da-fraction 0.8     # finetuned model attributions (80:20 DA split)
    python src/chrombpnet/modisco/launch.py --da-peaks tau                    # modisco on tau-specific DA peaks only
    python src/chrombpnet/modisco/launch.py --da-peaks pairwise               # modisco on pairwise DA peaks only
    python src/chrombpnet/modisco/launch.py --time 6-23:00:00 --mem 64G
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
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use attributions from finetuned models instead of base models",
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects attribution subdirectory (default: 0.5)",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=["tau", "pairwise"],
        default=None,
        help="Run modisco on DA-peaks-only attributions (default: all peaks)",
    )
    # Modisco parameters
    parser.add_argument(
        "-n",
        "--n-seqlets",
        type=int,
        default=1_000_000,
        help="max number of seqlets to use (default: 1_000_000)",
    )
    parser.add_argument(
        "-l",
        "--leiden",
        type=int,
        default=50,
        help="number of leiden clusters to use (default: 50)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=400,
        help="window to get seqlets from (default: 400)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2-00:00:00")
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]
    model_types = (
        args.model_type if args.model_type is not None else ["bias", "accessibility"]
    )

    fine_tag = "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    attr_subdir = f"chrombpnet_{fine_tag}" if args.finetuned else "chrombpnet"
    modisco_subdir = attr_subdir  # mirrors attribution directory naming
    da_suffix = f"_da_{args.da_peaks}" if args.da_peaks is not None else ""

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

    ohe_dir = REPO_ROOT / "attributions" / "chrombpnet"   # OHE always here
    attr_dir = REPO_ROOT / "attributions" / attr_subdir
    out_dir = REPO_ROOT / "modisco" / modisco_subdir
    log_dir = REPO_ROOT / "logs" / "chrombpnet_modisco"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_no_attr = 0
    for pseudobulk in pseudobulks:
        # OHE sequences file is always in attributions/chrombpnet/
        ohe_path = ohe_dir / f"{pseudobulk}_ohe.npz"
        if not ohe_path.exists():
            skipped_no_attr += len(model_types) * len(heads)
            continue

        for model_type in model_types:
            for head in heads:
                attr_path = attr_dir / f"{pseudobulk}_{model_type}_{head}{da_suffix}.npz"
                if not attr_path.exists():
                    skipped_no_attr += 1
                    continue

                # Skip if modisco output already exists
                out_h5 = out_dir / f"{pseudobulk}_{model_type}_{head}{da_suffix}.modisco.h5"
                if out_h5.exists():
                    skipped_done += 1
                    continue

                tag = f"_{fine_tag}" if args.finetuned else ""
                job_name = f"modisco{tag}{da_suffix}_{pseudobulk}_{model_type}_{head}"

                modisco_motifs_cmd = (
                    f"modisco motifs"
                    f" -s {ohe_path}"
                    f" -a {attr_path}"
                    f" -o {out_h5}"
                    f" -n {args.n_seqlets} -l {args.leiden} -w {args.window} -v"
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
                    time {modisco_motifs_cmd}
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
        f"\n{action} {submitted} jobs, skipped {skipped_no_attr} missing attributions, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
