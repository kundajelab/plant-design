#!/usr/bin/env python3
"""Submit SLURM jobs to train Cherimoya models for one or more pseudobulks/folds.

Each job trains one pseudobulk's selected folds sequentially in a bash loop
that re-checks each fold's completion marker at runtime (not just at
submission time), so a pre-empted, --requeue'd job (the akundaje/owners
partitions are preemptible) skips whatever folds finished before pre-emption
and only (re)trains the rest -- cherimoya trains fast enough that redoing a
partially-trained fold from scratch is cheap. "Completed" is judged by
`{pseudobulk}_f{fold}.final.torch`, which Cherimoya's fit() writes exactly
once at the very end of training -- `{pseudobulk}_f{fold}.torch` (no
`.final`) is the wrong marker, since it's overwritten throughout training
whenever validation correlation improves and can already exist after a
single epoch.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each pseudobulk's folds directly in
the foreground with inherited stdout/stderr, instead of submitting SLURM
jobs. It still uses Apptainer. SLURM resource flags
(--gpus/--partition/--cpus-per-task/--mem/--time) are ignored in this mode.
Combined with --dry-run, --local prints one runnable command per fold
instead of running them -- pipe that into something like
`simple_gpu_scheduler` to fan a GPU box you already have a shell on (e.g. an
Oak GPU node) out in parallel instead of running folds one at a time.

Usage:
    python src/cherimoya_pipeline/fit/launch.py                    # submit one job per pseudobulk, 5 folds each
    python src/cherimoya_pipeline/fit/launch.py --dry-run           # print sbatch scripts without submitting
    python src/cherimoya_pipeline/fit/launch.py --species A.thaliana --dataset base
    python src/cherimoya_pipeline/fit/launch.py -p bundle_sheath -f 0 -f 1
    python src/cherimoya_pipeline/fit/launch.py --time 48:00:00 --mem 32G
    python src/cherimoya_pipeline/fit/launch.py --local             # run in the foreground, no SLURM
    python src/cherimoya_pipeline/fit/launch.py --local --dry-run   # print one command per fold, e.g. for simple_gpu_scheduler
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap

from launcher_common import REPO_ROOT, SCRIPT_DIR, add_common_args, resolve_dataset_and_jobs

import species_config

TRAIN_SCRIPT = SCRIPT_DIR / "train_cherimoya.py"


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
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop after this many epochs with no validation improvement "
        "(default: disabled -- train the full --max-epochs).",
    )
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each pseudobulk's folds directly in the foreground, "
            "instead of submitting SLURM jobs; ignores SLURM resource "
            "flags (--gpus/--partition/--cpus-per-task/--mem/--time); "
            "combined with --dry-run, prints one command per fold instead "
            "of running them (e.g. to pipe into simple_gpu_scheduler)"
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
            "(default: the repo root -- add more if models/data live "
            "outside it, e.g. on scratch or oak)"
        ),
    )
    # SLURM resource flags
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
        help="SLURM GPU_SKU constraint (default: every SKU apptainer/check_gpu.py can test)",
    )
    parser.add_argument("--partition", type=str, default="akundaje,gpu,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument(
        "--time",
        type=str,
        default="24:00:00",
        help="time budget for the whole job, regardless of how many folds it trains",
    )
    return parser.parse_args()


def extra_train_args(args):
    parts = [
        "--n-filters",
        str(args.n_filters),
        "--n-layers",
        str(args.n_layers),
        "--max-epochs",
        str(args.max_epochs),
        "--negative-ratio",
        str(args.negative_ratio),
    ]
    if args.early_stopping is not None:
        parts += ["--early-stopping", str(args.early_stopping)]
    return parts


def main():
    args = parse_args()
    species, pseudobulks, folds = resolve_dataset_and_jobs(args)
    model_root = species_config.models_dir(species)
    if args.dataset != "base":
        model_root = model_root / args.dataset

    log_dir = species_config.logs_dir(species) / "cherimoya_fit"
    log_dir.mkdir(parents=True, exist_ok=True)

    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)
    train_args = extra_train_args(args)

    submitted = 0
    skipped_complete = 0
    local_failures = []

    for pseudobulk in pseudobulks:
        out_dir = model_root / pseudobulk

        # `.final.torch` is written exactly once, at the very end of
        # Cherimoya's fit() -- `.torch` (no `.final`) is overwritten
        # throughout training whenever validation correlation improves, so
        # checking that one would treat a fold killed after one epoch as
        # complete.
        folds_to_run = []
        for fold in folds:
            best_path = out_dir / f"{pseudobulk}_f{fold}.torch"
            final_path = out_dir / f"{pseudobulk}_f{fold}.final.torch"
            if args.rerun:
                best_path.unlink(missing_ok=True)
                final_path.unlink(missing_ok=True)
                folds_to_run.append(fold)
            elif not final_path.exists():
                folds_to_run.append(fold)
        skipped_complete += len(folds) - len(folds_to_run)
        if not folds_to_run:
            continue

        if args.local:
            for fold in folds_to_run:
                cmd = [
                    "apptainer",
                    "exec",
                    "--nv",
                    *[part for path in apptainer_binds for part in ("--bind", path)],
                    args.apptainer_image,
                    "python",
                    str(TRAIN_SCRIPT),
                    "--species",
                    species,
                    "--dataset",
                    args.dataset,
                    "-p",
                    pseudobulk,
                    "-f",
                    str(fold),
                    *train_args,
                    "-o",
                    str(out_dir),
                ]
                if args.dry_run:
                    print(shlex.join(cmd))
                    continue

                print(f"=== Training {pseudobulk} fold {fold} ===")
                result = subprocess.run(cmd, cwd=REPO_ROOT)
                if result.returncode != 0:
                    local_failures.append((pseudobulk, fold, result.returncode))
                    print(
                        f"ERROR: {pseudobulk} fold {fold} exited with "
                        f"{result.returncode}",
                        file=sys.stderr,
                    )
            submitted += 1
            continue

        job_name = f"cherimoya_fit_{args.dataset}_{pseudobulk}".replace(".", "p")
        folds_bash = " ".join(str(fold) for fold in folds_to_run)
        train_args_bash = " ".join(shlex.quote(part) for part in train_args)

        # Built as a flat list of lines (rather than a nested triple-quoted
        # block) and joined with the same 12-space indent as the rest of the
        # template below, so the final textwrap.dedent() still finds a
        # common prefix across every line -- a mismatched indent here leaves
        # the whole script indented, which sbatch rejects as "not a batch
        # script".
        setup_lines = [
            f"TRAIN_SCRIPT={shlex.quote(str(TRAIN_SCRIPT))}",
            f'APPTAINER_IMAGE="{args.apptainer_image}"',
            f"PSEUDOBULK={shlex.quote(pseudobulk)}",
            f"OUT_DIR={shlex.quote(str(out_dir))}",
            f"FOLDS=({folds_bash})",
            "",
        ]
        setup_block = "\n            ".join(setup_lines)
        run_fold_cmd = (
            f'apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" '
            f'python "$TRAIN_SCRIPT" --species {shlex.quote(species)} '
            f'--dataset {shlex.quote(args.dataset)} -p "$PSEUDOBULK" -f "$FOLD" '
            f'{train_args_bash} -o "$OUT_DIR"'
        )

        # The fold loop re-checks each fold's .final.torch at runtime (not
        # just once at submission time, in `folds_to_run` above), so a
        # SLURM-requeued rerun of this same script skips whatever folds
        # finished before pre-emption instead of blindly retraining all of
        # them. --requeue is what makes a pre-empted job actually retry
        # automatically; --open-mode=append keeps prior attempts' output
        # instead of truncating the log on each retry.
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
            #SBATCH --requeue
            #SBATCH --open-mode=append
            #SBATCH --output={log_dir}/{job_name}.out
            #SBATCH --error={log_dir}/{job_name}.err

            {setup_block}
            nvidia-smi -L

            for FOLD in "${{FOLDS[@]}}"; do
                FINAL_PATH="$OUT_DIR/${{PSEUDOBULK}}_f${{FOLD}}.final.torch"
                if [[ -f "$FINAL_PATH" ]]; then
                    echo "Skipping already-completed fold $FOLD: $FINAL_PATH"
                    continue
                fi
                echo "Training $PSEUDOBULK fold $FOLD"
                {run_fold_cmd}
            done
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
        unit = "pseudobulks"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "jobs"
    total = len(pseudobulks) * len(folds)
    print(
        f"\n{action} {submitted} {unit}, skipped {skipped_complete} "
        f"already-trained fold(s) ({total} total folds)"
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} fold(s) failed:")
            for pseudobulk, fold, returncode in local_failures:
                print(f"  {pseudobulk} fold {fold} (exit {returncode})")
        else:
            print("All folds completed successfully.")


if __name__ == "__main__":
    main()
