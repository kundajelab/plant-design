#!/usr/bin/env python3
"""Submit SLURM jobs to run DeepLIFT/SHAP attributions for Cherimoya models.

Reads pseudobulks from configs/<species>/pseudobulks.txt and submits one
sbatch job per (pseudobulk, head) pair via `deepliftshap.py`.

Jobs are skipped if the attribution output already exists, if the required
loci BED file is missing (run filter_nonACGT_regions.py, differential_accessibility.py,
or write_pairwise_higher_beds.py as needed), or if the required model files are
missing (run train_cherimoya.py first).

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each (pseudobulk, head) job directly
in the foreground with inherited stdout/stderr, instead of submitting SLURM
jobs. Combined with --dry-run, --local prints one runnable command per job
instead of running them -- pipe that into something like
`simple_gpu_scheduler`.

Usage:
    python src/cherimoya_pipeline/attribute/launch.py                                    # all pseudobulks, both heads
    python src/cherimoya_pipeline/attribute/launch.py --dry-run                          # print sbatch scripts without submitting
    python src/cherimoya_pipeline/attribute/launch.py --head count                       # count head only
    python src/cherimoya_pipeline/attribute/launch.py --finetuned                        # finetuned models (50:50 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --finetuned --da-fraction 0.8     # finetuned models (80:20 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --adapter                         # default daicon adapters (50:50 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --adapter --da-fraction 0.8       # default daicon adapters (80:20 DA split)
    python src/cherimoya_pipeline/attribute/launch.py --da-peaks tau                    # attributions on tau-specific peaks only
    python src/cherimoya_pipeline/attribute/launch.py --da-peaks specific_union       # attributions on per-cell-type union DA peaks
    python src/cherimoya_pipeline/attribute/launch.py --time 24:00:00 --mem 128G
    python src/cherimoya_pipeline/attribute/launch.py --local --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from da_peaks import (
    DATASETS,
    DA_PEAK_CHOICES,
    da_suffix,
    model_dir as dataset_model_dir,
    read_pseudobulks,
    resolve_loci_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "cherimoya_pipeline" / "benchmark"))

import species_config  # noqa: E402
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402

N_FOLDS = 5


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run attributions on DA peaks instead of all cell-type peaks (default: all peaks)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to use (default: base).",
    )
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned models instead of the base model",
        adapter_help="Use default daicon adapter bundles for the selected DA fraction",
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
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="48:00:00")
    args = parser.parse_args()
    species = species_config.resolve_species(args.species)
    variant = resolve_model_variant(args)
    fine_tag = variant["fine_tag"]
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]

    heads = args.head if args.head is not None else ["profile", "count"]

    if adapter_id is not None:
        base_subdir = f"cherimoya_adapter_{adapter_id}"
    else:
        base_subdir = f"cherimoya_{fine_tag}" if args.finetuned else "cherimoya"
    out_subdir = base_subdir if args.dataset == "base" else f"{base_subdir}_{args.dataset}"

    if args.da_peaks is not None and args.dataset != "base":
        print(
            "Error: --da-peaks is currently only defined for --dataset base.",
            file=sys.stderr,
        )
        sys.exit(1)

    pseudobulks = read_pseudobulks(species, args.dataset)

    attr_dir = species_config.attributions_dir(species)
    log_dir = species_config.logs_dir(species) / "cherimoya_deepliftshap"
    log_dir.mkdir(parents=True, exist_ok=True)
    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)

    submitted = 0
    skipped_done = 0
    skipped_missing = 0
    local_failures = []

    for pseudobulk in pseudobulks:
        peaks_path = resolve_loci_path(
            species, pseudobulk, args.da_peaks, dataset=args.dataset
        )
        if not peaks_path.exists():
            skipped_missing += len(heads)
            print(
                f"SKIP {pseudobulk}: loci file missing ({peaks_path})",
                file=sys.stderr,
            )
            continue

        model_dir = dataset_model_dir(species, args.dataset, pseudobulk) / model_subdir
        model_paths = [
            model_dir / f"{pseudobulk}_f{fold}.torch" for fold in range(N_FOLDS)
        ]
        missing_models = [p for p in model_paths if not p.exists()]
        if missing_models:
            skipped_missing += len(heads)
            print(
                f"SKIP {pseudobulk}: {len(missing_models)} model file(s) missing",
                file=sys.stderr,
            )
            continue
        if adapter_id is not None:
            adapter_paths = [
                dataset_model_dir(species, args.dataset, pseudobulk)
                / "adapters"
                / adapter_id
                / f"{pseudobulk}_f{fold}.adapter.pt"
                for fold in range(N_FOLDS)
            ]
            missing_adapters = [p for p in adapter_paths if not p.exists()]
            if missing_adapters:
                skipped_missing += len(heads)
                print(
                    f"SKIP {pseudobulk}: {len(missing_adapters)} adapter bundle(s) missing",
                    file=sys.stderr,
                )
                continue

        for head in heads:
            suffix = da_suffix(args.da_peaks)
            out_path = attr_dir / out_subdir / f"{pseudobulk}_{head}{suffix}.npz"
            if out_path.exists():
                skipped_done += 1
                continue

            tag = (
                f"_adapter_{adapter_id}"
                if adapter_id is not None
                else (f"_{fine_tag}" if args.finetuned else "")
            )
            da_tag = da_suffix(args.da_peaks)
            job_name = f"deepliftshap{tag}{da_tag}_{pseudobulk}_{head}".replace(".", "p")
            script_path = Path(__file__).resolve().parent / "deepliftshap.py"

            finetuned_flags = (
                ["--finetuned", "--da-fraction", str(args.da_fraction)]
                if args.finetuned
                else []
            )
            if adapter_id is None:
                adapter_flags = []
            elif args.adapter:
                adapter_flags = ["--adapter", "--da-fraction", str(args.da_fraction)]
            else:
                adapter_flags = ["--adapter-sweep-id", adapter_id]
            da_peaks_flag = ["--da-peaks", args.da_peaks] if args.da_peaks is not None else []
            command = [
                "python",
                str(script_path),
                "--species",
                species,
                "-p",
                pseudobulk,
                "--dataset",
                args.dataset,
                "--head",
                head,
                "-b",
                str(args.batch_size),
                "-v",
                *finetuned_flags,
                *adapter_flags,
                *da_peaks_flag,
            ]

            if args.local:
                full_cmd = [
                    "apptainer",
                    "exec",
                    "--nv",
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
            deepliftshap_cmd = (
                f'apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" {command_bash}'
            )

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

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
    total = len(pseudobulks) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_missing} missing inputs, "
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
