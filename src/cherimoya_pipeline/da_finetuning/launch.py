#!/usr/bin/env python3
"""Submit one SLURM job per pseudobulk to run standard DA fine-tuning.

This launcher is for the small set of named runs used downstream:

- full-model fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter fine-tuning at 0.5 and 0.8 DA fractions

Use ``launch_adapter_sweep.py`` for the full adapter hyperparameter sweep.

Each job runs one pseudobulk's full set of steps (every selected run group x
fold x DA-fraction combination) sequentially in a bash loop that re-checks
each step's completion marker (its training log) at runtime, not just at
submission time -- mirrors fit/launch.py's per-pseudobulk fold loop, so a
pre-empted, --requeue'd job (the akundaje/owners partitions are preemptible)
skips whatever steps finished before pre-emption and only (re)runs the rest.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each job's steps directly in the
foreground with inherited stdout/stderr, instead of submitting SLURM jobs.
Combined with --dry-run, --local prints one runnable command per step
instead of running them.

--print-commands prints every step's apptainer command to stdout (no SLURM
submission, no job bundling, no skipping of already-finished steps) and
exits -- for hand-picking commands to run on an interactive allocation.
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASETS = ("base", "fine", "coarse")
N_FOLDS = 5
DA_FRACTIONS = (0.5, 0.8)

RUN_GROUPS = ("full", "adapter")
DEFAULT_ADAPTER_LR = 1e-3


def da_tag(da_fraction):
    return str(da_fraction).replace(".", "p")


def full_tag(da_fraction):
    return "finetuned" if da_fraction == 0.5 else f"finetuned_{da_fraction:g}"


def default_adapter_run_id(da_fraction):
    return f"daicon_default_da{da_tag(da_fraction)}"


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def model_root(species, dataset):
    root = species_config.models_dir(species)
    return root if dataset == "base" else root / dataset


def load_pseudobulks(species, dataset, selected):
    if selected is not None:
        return selected
    return species_config.read_pseudobulks(species, dataset)


def common_training_args(args):
    common = [
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--max-jitter",
        str(args.max_jitter),
        "--negative-ratio",
        str(args.negative_ratio),
    ]
    if args.blacklist is not None:
        common += ["--blacklist", str(args.blacklist)]
    return common


def build_full_job(args, species, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "cherimoya_pipeline" / "da_finetuning" / "finetune_da.py"
    command = [
        "python",
        str(script),
        "--species",
        species,
        "--dataset",
        args.dataset,
        "-p",
        pseudobulk,
        "-f",
        str(fold),
        "--mode",
        "full",
        "--da-fraction",
        str(da_fraction),
        *common_training_args(args),
        "-v",
    ]
    finish_path = (
        model_root(species, args.dataset)
        / pseudobulk
        / full_tag(da_fraction)
        / f"{pseudobulk}_f{fold}.log"
    )
    return {
        "run_group": "full",
        "run_id": full_tag(da_fraction),
        "fold": fold,
        "command": command,
        "finish_path": finish_path,
    }


def build_adapter_job(args, species, pseudobulk, fold, da_fraction):
    script = REPO_ROOT / "src" / "cherimoya_pipeline" / "da_finetuning" / "finetune_da.py"
    run_id = default_adapter_run_id(da_fraction)
    command = [
        "python",
        str(script),
        "--species",
        species,
        "--dataset",
        args.dataset,
        "-p",
        pseudobulk,
        "-f",
        str(fold),
        "--mode",
        "adapter",
        "--da-fraction",
        str(da_fraction),
        *common_training_args(args),
        "--lr",
        str(args.adapter_lr),
        "-v",
    ]
    finish_path = (
        model_root(species, args.dataset)
        / pseudobulk
        / "adapters"
        / run_id
        / f"{pseudobulk}_f{fold}.log.tsv"
    )
    return {
        "run_group": "adapter",
        "run_id": run_id,
        "fold": fold,
        "command": command,
        "finish_path": finish_path,
    }


def build_jobs(args, species):
    """Return {pseudobulk: [step, ...]} -- every selected run group x fold x
    DA-fraction combination for that pseudobulk, all bundled into one job.
    """
    pseudobulks = load_pseudobulks(species, args.dataset, args.pseudobulk)
    folds = args.fold if args.fold is not None else list(range(N_FOLDS))
    da_fractions = args.da_fraction if args.da_fraction is not None else list(DA_FRACTIONS)

    builders = {
        "full": build_full_job,
        "adapter": build_adapter_job,
    }
    jobs = {}
    for pseudobulk in pseudobulks:
        entries = []
        for fold in folds:
            for run_group in args.run:
                for da_fraction in da_fractions:
                    entries.append(
                        builders[run_group](args, species, pseudobulk, fold, da_fraction)
                    )
        jobs[pseudobulk] = entries
    return jobs


def print_finished_report(jobs):
    total_finished = 0
    total_expected = 0
    for pseudobulk, entries in jobs.items():
        finished = sum(entry["finish_path"].exists() for entry in entries)
        total_finished += finished
        total_expected += len(entries)
        status = "DONE" if finished == len(entries) else "PENDING"
        print(f"{status}\t{pseudobulk}\t{finished}/{len(entries)} training logs")
        if finished != len(entries):
            for entry in entries:
                if not entry["finish_path"].exists():
                    print(
                        f"MISSING\t{entry['run_group']}:{entry['run_id']}"
                        f"\tfold={entry['fold']}\t{entry['finish_path']}"
                    )
    print(f"\nFinished {total_finished}/{total_expected} expected training logs.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--run",
        action="append",
        choices=RUN_GROUPS,
        default=None,
        help="Run group to submit; repeatable. Default: all groups.",
    )
    parser.add_argument("--pseudobulk", action="append", default=None)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to finetune (default: base).",
    )
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--da-fraction", type=float, action="append", default=None)
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=DEFAULT_ADAPTER_LR,
        help=f"Learning rate for adapter fine-tuning (default: {DEFAULT_ADAPTER_LR:g}). "
        "Full-model fine-tuning uses finetune_da.py's own "
        "--muon-lr/--adam-lr/--lw-lr defaults instead.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-jitter", type=int, default=500)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument(
        "--blacklist",
        type=str,
        default=None,
        help="Optional BED of regions to exclude",
    )
    parser.add_argument(
        "--check-finished",
        action="store_true",
        help="report expected training logs and exit",
    )
    parser.add_argument(
        "--rerun-finished",
        action="store_true",
        help="delete existing training logs and rerun every step",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
    )
    parser.add_argument(
        "--print-commands",
        action="store_true",
        help=(
            "print every step's apptainer command to stdout and exit -- "
            "unlike --local --dry-run, this does not skip steps whose "
            "training log already exists, so you get the full list to "
            "hand-pick from for running on an interactive allocation "
            "(no SLURM submission, no job bundling)"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each job's steps directly in the foreground, instead of "
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
    parser.add_argument("--partition", type=str, default="akundaje,gpu,owners")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument(
        "--time",
        type=str,
        default="4-00:00:00",
        help="time budget for the whole job, regardless of how many steps it runs",
    )
    args = parser.parse_args()

    if args.run is None:
        args.run = list(RUN_GROUPS)
    species = species_config.resolve_species(args.species)
    choices = load_pseudobulks(species, args.dataset, None)
    selected = args.pseudobulk or choices
    unknown = [p for p in selected if p not in choices]
    if unknown:
        print(
            f"Error: unknown pseudobulk(s) for species '{species}', dataset "
            f"'{args.dataset}': {', '.join(unknown)}. Choose one of: {', '.join(choices)}",
            file=sys.stderr,
        )
        sys.exit(1)
    args.pseudobulk = selected

    jobs = build_jobs(args, species)
    if args.check_finished:
        print_finished_report(jobs)
        return

    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)
    apptainer_image_arg = shlex.quote(args.apptainer_image)

    if args.print_commands:
        for entries in jobs.values():
            for entry in entries:
                full_cmd = [
                    "apptainer",
                    "exec",
                    "--nv",
                    *[part for path in apptainer_binds for part in ("--bind", path)],
                    args.apptainer_image,
                    *entry["command"],
                ]
                print(shlex.join(full_cmd))
        return

    log_dir = species_config.logs_dir(species) / "cherimoya_da_finetuning"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_empty = 0
    skipped_steps = 0
    local_failures = []

    for pseudobulk, entries in jobs.items():
        if args.rerun_finished:
            for entry in entries:
                entry["finish_path"].unlink(missing_ok=True)

        skipped_steps += sum(entry["finish_path"].exists() for entry in entries)
        if all(entry["finish_path"].exists() for entry in entries):
            skipped_empty += 1
            continue

        job_name = f"da_ft_{args.dataset}_{pseudobulk}".replace(".", "p")

        if args.local:
            for entry in entries:
                if entry["finish_path"].exists():
                    continue
                full_cmd = [
                    "apptainer",
                    "exec",
                    "--nv",
                    *[part for path in apptainer_binds for part in ("--bind", path)],
                    args.apptainer_image,
                    *entry["command"],
                ]
                label = f"{entry['run_group']}:{entry['run_id']} fold {entry['fold']}"
                if args.dry_run:
                    print(shlex.join(full_cmd))
                    continue
                print(f"=== {job_name}: {label} ===")
                result = subprocess.run(full_cmd, cwd=REPO_ROOT)
                if result.returncode != 0:
                    local_failures.append((job_name, label, result.returncode))
                    print(f"ERROR: step exited with {result.returncode}", file=sys.stderr)
            submitted += 1
            continue

        # Every selected step is embedded in the script (not just the ones
        # still pending at submission time), each guarded by a runtime check
        # of its own training-log completion marker -- mirrors
        # fit/launch.py's fold loop, so a --requeue'd rerun after
        # pre-emption skips whatever steps finished before it was killed
        # instead of blindly redoing all of them.
        # Every line of every step is collected into one flat list and
        # joined with the same 12-space indent as the rest of the template
        # below, so the final textwrap.dedent() still finds a common prefix
        # across every line -- a mismatched indent here (e.g. joining
        # multi-line blocks at the top level only) leaves the whole script
        # indented, which sbatch rejects as "not a batch script".
        step_lines = []
        for entry in entries:
            finish_path_bash = shlex.quote(str(entry["finish_path"]))
            label = f"{entry['run_group']}:{entry['run_id']} fold {entry['fold']}"
            run_cmd = (
                f"apptainer exec --nv {apptainer_bind_args} {apptainer_image_arg} "
                + " ".join(shlex.quote(part) for part in entry["command"])
            )
            step_lines += [
                f"if [[ -f {finish_path_bash} ]]; then",
                f'    echo "Skipping already-finished step: {label}"',
                "else",
                f'    echo "Running step: {label}"',
                f"    time {run_cmd}",
                "fi",
                "",
            ]
        run_block = "\n            ".join(step_lines)

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

            nvidia-smi -L

            {run_block}
        """)

        if args.dry_run:
            print(f"--- {job_name} ({len(entries)} step(s)) ---")
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
            print(f"ERROR submitting {job_name}: {result.stderr.strip()}", file=sys.stderr)

    if args.local:
        action = "Would run" if args.dry_run else "Ran"
        unit = "pseudobulk job(s)"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "sbatch job(s)"
    print(
        f"\n{action} {submitted} {unit}; skipped {skipped_empty} fully-finished "
        f"pseudobulk(s), {skipped_steps} already-finished step(s) (skipped at runtime)."
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} step(s) failed:")
            for job_name, label, returncode in local_failures:
                print(f"  {job_name}: {label} (exit {returncode})")
        else:
            print("All steps completed successfully.")


if __name__ == "__main__":
    main()
