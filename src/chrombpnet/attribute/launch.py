#!/usr/bin/env python3
"""Submit SLURM jobs to run DeepLIFT/SHAP attributions for ChromBPNet models.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, model_type, head) triple via `deepliftshap.py`.

Jobs are skipped if the attribution output already exists, if the required
loci BED file is missing (run filter_nonACGT_regions.py, differential_accessibility.py,
or write_pairwise_higher_beds.py as needed), or if the required model files are
missing (run convert_to_torch.py first).

Usage:
    python src/chrombpnet/attribute/launch.py                                    # all pseudobulks, both model types, both heads
    python src/chrombpnet/attribute/launch.py --dry-run                          # print sbatch scripts without submitting
    python src/chrombpnet/attribute/launch.py --head count                       # count head only
    python src/chrombpnet/attribute/launch.py --model-type bias                  # bias model only
    python src/chrombpnet/attribute/launch.py --finetuned                        # finetuned models (50:50 DA split)
    python src/chrombpnet/attribute/launch.py --finetuned --da-fraction 0.8     # finetuned models (80:20 DA split)
    python src/chrombpnet/attribute/launch.py --da-peaks tau                    # attributions on tau-specific peaks only
    python src/chrombpnet/attribute/launch.py --da-peaks specific_union       # attributions on per-cell-type union DA peaks
    python src/chrombpnet/attribute/launch.py --time 24:00:00 --mem 128G
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from da_peaks import DA_PEAK_CHOICES, da_suffix, resolve_loci_path

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
        "--model",
        type=str,
        action="append",
        choices=["bias", "accessibility"],
        default=None,
        metavar="MODEL_TYPE",
        help="model type(s) to run; repeatable (default: bias accessibility)",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run attributions on DA peaks instead of all cell-type peaks (default: all peaks)",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use finetuned models instead of base torch/ models",
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects model/output subdirectory (default: 0.5)",
    )
    parser.add_argument(
        "--adapter-sweep-id",
        type=str,
        default=None,
        help="Use adapter bundles from models/{pseudobulk}/adapters/{adapter_sweep_id}/",
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
        "--constraint", type=str, default="GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR"
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="48:00:00")
    args = parser.parse_args()

    if args.finetuned and args.adapter_sweep_id is not None:
        print("Error: use either --finetuned or --adapter-sweep-id, not both.", file=sys.stderr)
        sys.exit(1)

    heads = args.head if args.head is not None else ["profile", "count"]
    model_types = (
        args.model_type if args.model_type is not None else ["bias", "accessibility"]
    )

    fine_tag = "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    model_subdir = fine_tag if args.finetuned else "torch"
    if args.adapter_sweep_id is not None:
        out_subdir = f"chrombpnet_adapter_{args.adapter_sweep_id}"
    else:
        out_subdir = f"chrombpnet_{fine_tag}" if args.finetuned else "chrombpnet"

    if (args.finetuned or args.adapter_sweep_id is not None) and "bias" in model_types:
        print(
            "Warning: bias model is not available for finetuned/adapter runs; skipping bias model type.",
            file=sys.stderr,
        )
        model_types = [m for m in model_types if m != "bias"]
        if not model_types:
            sys.exit(0)

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
        peaks_path = resolve_loci_path(REPO_ROOT, pseudobulk, args.da_peaks)
        if not peaks_path.exists():
            skipped_missing += len(model_types) * len(heads)
            print(
                f"SKIP {pseudobulk}: loci file missing ({peaks_path})",
                file=sys.stderr,
            )
            continue

        for model_type in model_types:
            model_dir = REPO_ROOT / "models" / pseudobulk / model_subdir
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
            if args.adapter_sweep_id is not None:
                adapter_paths = [
                    REPO_ROOT
                    / "models"
                    / pseudobulk
                    / "adapters"
                    / args.adapter_sweep_id
                    / f"{pseudobulk}_f{fold}.adapter.pt"
                    for fold in range(N_FOLDS)
                ]
                missing_adapters = [p for p in adapter_paths if not p.exists()]
                if missing_adapters:
                    skipped_missing += len(heads)
                    print(
                        f"SKIP {pseudobulk}/{model_type}: {len(missing_adapters)} adapter bundle(s) missing",
                        file=sys.stderr,
                    )
                    continue

            for head in heads:
                suffix = da_suffix(args.da_peaks)
                out_path = attr_dir / out_subdir / f"{pseudobulk}_{model_type}_{head}{suffix}.npz"
                if out_path.exists():
                    skipped_done += 1
                    continue

                tag = (
                    f"_adapter_{args.adapter_sweep_id}"
                    if args.adapter_sweep_id is not None
                    else (f"_{fine_tag}" if args.finetuned else "")
                )
                da_tag = da_suffix(args.da_peaks)
                job_name = f"deepliftshap{tag}{da_tag}_{pseudobulk}_{model_type}_{head}"
                script_path = Path(__file__).resolve().parent / "deepliftshap.py"

                finetuned_flags = (
                    f" --finetuned --da-fraction {args.da_fraction}" if args.finetuned else ""
                )
                adapter_flags = (
                    f" --adapter-sweep-id {args.adapter_sweep_id}"
                    if args.adapter_sweep_id is not None
                    else ""
                )
                da_peaks_flag = f" --da-peaks {args.da_peaks}" if args.da_peaks is not None else ""
                deepliftshap_cmd = (
                    f"python {script_path}"
                    f" -p {pseudobulk}"
                    f" -m {model_type}"
                    f" --head {head}"
                    f" -b {args.batch_size}"
                    f" -v"
                    f"{finetuned_flags}"
                    f"{adapter_flags}"
                    f"{da_peaks_flag}"
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
