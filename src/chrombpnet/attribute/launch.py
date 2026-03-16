#!/usr/bin/env python3
"""Submit SLURM jobs to run DeepLIFT/SHAP attributions for ChromBPNet models.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, model_type, head) triple via `deepliftshap.py`.

Jobs are skipped if the attribution output already exists, if the required
peaks_ACGT file is missing (run filter_nonACGT_regions.py first), or if the
required model files are missing (run convert_to_torch.py first).

Usage:
    python src/chrombpnet/attribute/launch.py                          # all pseudobulks, both model types, both heads
    python src/chrombpnet/attribute/launch.py --dry-run                # print sbatch scripts without submitting
    python src/chrombpnet/attribute/launch.py --head count             # count head only
    python src/chrombpnet/attribute/launch.py --model-type bias        # bias model only
    python src/chrombpnet/attribute/launch.py --time 24:00:00 --mem 128G
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS_PATH = REPO_ROOT / "configs" / "pseudobulks.txt"
N_FOLDS = 5


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
    # deepliftshap parameters
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="batch size for attribution (default: 64)",
    )
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="gpu,akundaje,owners")
    parser.add_argument(
        "--constraint",
        type=str,
        default="GPU_SKU:A100_SXM4|GPU_SKU:L40S|GPU_SKU:H100_SXM5|GPU_SKU:RTX_3090",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="48:00:00")
    args = parser.parse_args()

    heads = args.head if args.head is not None else ["profile", "count"]
    model_types = (
        args.model_type if args.model_type is not None else ["bias", "accessibility"]
    )

    # Load pseudobulk list
    pseudobulks = PSEUDOBULKS_PATH.read_text().splitlines()
    pseudobulks = [p for p in pseudobulks if p.strip()]

    attr_dir = REPO_ROOT / "attributions"
    log_dir = REPO_ROOT / "logs" / "chrombpnet_deepliftshap"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0

    for pseudobulk in pseudobulks:
        peaks_path = REPO_ROOT / "data" / "peaks" / f"{pseudobulk}_peaks_ACGT.narrowPeak"
        if not peaks_path.exists():
            skipped_missing += len(model_types) * len(heads)
            print(
                f"SKIP {pseudobulk}: peaks_ACGT file missing ({peaks_path})",
                file=sys.stderr,
            )
            continue

        for model_type in model_types:
            model_dir = REPO_ROOT / "models" / pseudobulk / "torch"
            model_paths = [
                model_dir / f"{pseudobulk}_f{fold}.{model_type}.torch"
                for fold in range(N_FOLDS)
            ]
            missing_models = [p for p in model_paths if not p.exists()]
            if missing_models:
                skipped_missing += len(heads)
                print(
                    f"SKIP {pseudobulk}/{model_type}: {len(missing_models)} model file(s) missing",
                    file=sys.stderr,
                )
                continue

            for head in heads:
                out_path = attr_dir / f"{pseudobulk}_{model_type}_{head}.npz"
                if out_path.exists():
                    skipped_done += 1
                    continue

                job_name = f"deepliftshap_{pseudobulk}_{model_type}_{head}"
                script_path = Path(__file__).resolve().parent / "deepliftshap.py"

                deepliftshap_cmd = (
                    f"python {script_path}"
                    f" -p {pseudobulk}"
                    f" -m {model_type}"
                    f" --head {head}"
                    f" -b {args.batch_size}"
                    f" -v"
                )

                sbatch_script = textwrap.dedent(f"""\
                    #!/bin/bash -l
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
                    nvidia-smi -L

                    time {deepliftshap_cmd}
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
        f"\n{action} {submitted} jobs, skipped {skipped_missing} missing inputs, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
