#!/usr/bin/env python3
"""Submit one SLURM job per model variant to benchmark active Cherimoya models.

This launcher covers the main reporting path:

- base Cherimoya
- full-model DA fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter DA fine-tuning at 0.5 and 0.8 DA fractions

Development-only adapter parameter sweeps are handled separately by
``benchmark_lora_sweep.py``.

Each job runs one variant's full set of steps (cross-pseudobulk benchmark.py
calls plus benchmark_da.py and, for the base dataset,
pairwise_correlations_predicted.py) sequentially in a bash loop that
re-checks each step's own output-file completion marker at runtime, not just
at submission time -- mirrors fit/launch.py's per-pseudobulk fold loop, so a
pre-empted, --requeue'd job (the akundaje/owners partitions are preemptible)
skips whatever steps finished before pre-emption and only (re)runs the rest.
These steps are individually quick, so one job per pseudobulk would mostly
be SLURM scheduling overhead.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each variant's steps directly in the
foreground with inherited stdout/stderr, instead of submitting SLURM jobs.
Combined with --dry-run, --local prints one runnable command per step
instead of running them -- pipe that into something like
`simple_gpu_scheduler`.

--print-commands prints every not-yet-finished step's command to stdout
prefixed with `uv run` (no apptainer wrapper, unlike --local --dry-run, and
no SLURM submission or job bundling) and exits -- for hand-picking commands
to run directly on a cluster where `plant-design` has been `uv sync --extra
cherimoya`'d (e.g. an Oak GPU node), or inside a shell you've already
entered the Apptainer image from.

Usage:
    python src/cherimoya_pipeline/benchmark/launch.py
    python src/cherimoya_pipeline/benchmark/launch.py --dry-run
    python src/cherimoya_pipeline/benchmark/launch.py --variant base --variant finetuned
    python src/cherimoya_pipeline/benchmark/launch.py --local
    python src/cherimoya_pipeline/benchmark/launch.py --local --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

BENCHMARK_DIR = REPO_ROOT / "src" / "cherimoya_pipeline" / "benchmark"
DATASETS = ("base", "fine", "coarse")


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    flags: tuple[str, ...]
    # Paths below are relative to performance_metrics_dir(species).
    cross_dir: Path
    da_path: Path
    pairwise_path: Path


def adapter_variant(name, label, run_id, da_fraction):
    return Variant(
        name=name,
        label=label,
        flags=("--adapter", "--da-fraction", str(da_fraction)),
        cross_dir=Path("adapters") / run_id,
        da_path=Path("differential_accessibility") / f"da_benchmark_adapter_{run_id}.json",
        pairwise_path=Path("adapters")
        / run_id
        / f"pairwise_correlations_predicted_adapter_{run_id}.json",
    )


VARIANTS = {
    "base": Variant(
        name="base",
        label="Base",
        flags=(),
        cross_dir=Path("cherimoya"),
        da_path=Path("differential_accessibility") / "da_benchmark.json",
        pairwise_path=Path("pairwise_correlations_predicted.json"),
    ),
    "finetuned": Variant(
        name="finetuned",
        label="Finetuned 50:50",
        flags=("--finetuned",),
        cross_dir=Path("cherimoya_finetuned"),
        da_path=Path("differential_accessibility") / "da_benchmark_finetuned.json",
        pairwise_path=Path("pairwise_correlations_predicted_finetuned.json"),
    ),
    "finetuned_0.8": Variant(
        name="finetuned_0.8",
        label="Finetuned 80:20",
        flags=("--finetuned", "--da-fraction", "0.8"),
        cross_dir=Path("cherimoya_finetuned_0.8"),
        da_path=Path("differential_accessibility") / "da_benchmark_finetuned_0.8.json",
        pairwise_path=Path("pairwise_correlations_predicted_finetuned_0.8.json"),
    ),
    "adapter": adapter_variant(
        "adapter",
        "Adapter 50:50",
        "daicon_default_da0p5",
        0.5,
    ),
    "adapter_0.8": adapter_variant(
        "adapter_0.8",
        "Adapter 80:20",
        "daicon_default_da0p8",
        0.8,
    ),
}


def default_apptainer_image():
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def slurm_name(text):
    return text.replace("/", "_").replace(".", "p").replace("+", "_").replace("-", "_")


def dataset_suffix(dataset_name):
    return "" if dataset_name == "base" else f"_{dataset_name}"


def cross_target_path(species, dataset_name, variant, pseudobulk):
    metrics_root = species_config.performance_metrics_dir(species)
    metrics_subdir = (
        variant.cross_dir
        if dataset_name == "base"
        else Path(f"{variant.cross_dir}_{dataset_name}")
    )
    return metrics_root / metrics_subdir / f"{pseudobulk}.json"


def da_target_path(species, dataset_name, variant):
    metrics_root = species_config.performance_metrics_dir(species)
    if dataset_name == "base":
        return metrics_root / variant.da_path
    return (
        metrics_root
        / f"differential_accessibility{dataset_suffix(dataset_name)}"
        / variant.da_path.name
    )


def benchmark_da_peaks_path(da_path):
    # benchmark_da.py's --save-output always writes this as a .csv
    # regardless of da_path's own (.json) extension.
    return da_path.with_name(
        da_path.stem.replace("da_benchmark", "da_benchmark_peaks") + ".csv"
    )


def da_finish_path(species, dataset_name, variant, save_da_outputs):
    da_path = da_target_path(species, dataset_name, variant)
    return benchmark_da_peaks_path(da_path) if save_da_outputs else da_path


def pairwise_target_path(species, variant):
    return species_config.performance_metrics_dir(species) / variant.pairwise_path


def build_variant_entries(args, species, variant):
    """Return every step for one variant -- cross-pseudobulk benchmark.py
    calls, benchmark_da.py, and (base dataset only)
    pairwise_correlations_predicted.py -- each tagged with the output path
    that marks it complete.
    """
    entries = []

    for pseudobulk in args.pseudobulks:
        cmd = [
            "python",
            str(BENCHMARK_DIR / "benchmark.py"),
            "--species",
            species,
            "--dataset",
            args.dataset,
            "-p",
            pseudobulk,
            "-b",
            str(args.batch_size),
            *variant.flags,
        ]
        if args.save_cross_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        entries.append(
            {
                "label": f"cross:{pseudobulk}",
                "command": cmd,
                "finish_path": cross_target_path(species, args.dataset, variant, pseudobulk),
            }
        )

    da_cmd = [
        "python",
        str(BENCHMARK_DIR / "benchmark_da.py"),
        "--species",
        species,
        "--dataset",
        args.dataset,
        "-b",
        str(args.batch_size),
        *variant.flags,
    ]
    if args.save_da_outputs:
        da_cmd.append("--save-output")
    if args.verbose:
        da_cmd.append("-v")
    entries.append(
        {
            "label": "da",
            "command": da_cmd,
            "finish_path": da_finish_path(species, args.dataset, variant, args.save_da_outputs),
        }
    )

    if args.dataset == "base":
        entries.append(
            {
                "label": "pairwise",
                "command": [
                    "python",
                    str(BENCHMARK_DIR / "pairwise_correlations_predicted.py"),
                    "--species",
                    species,
                    *variant.flags,
                ],
                "finish_path": pairwise_target_path(species, variant),
            }
        )

    return entries


def build_jobs(args, species):
    """Return {variant_name: [step, ...]} -- every step for that variant,
    all bundled into one job.
    """
    variant_names = args.variant or list(VARIANTS)
    return {name: build_variant_entries(args, species, VARIANTS[name]) for name in variant_names}


def print_finished_report(jobs):
    total_finished = 0
    total_expected = 0
    for variant_name, entries in jobs.items():
        finished = sum(entry["finish_path"].exists() for entry in entries)
        total_finished += finished
        total_expected += len(entries)
        status = "DONE" if finished == len(entries) else "PENDING"
        print(f"{status}\t{variant_name}\t{finished}/{len(entries)} outputs")
        if finished != len(entries):
            for entry in entries:
                if not entry["finish_path"].exists():
                    print(f"MISSING\t{entry['label']}\t{entry['finish_path']}")
    print(f"\nFinished {total_finished}/{total_expected} expected outputs.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        default=None,
        help="Model variant to benchmark; repeatable. Default: all active variants.",
    )
    parser.add_argument("--pseudobulk", dest="pseudobulks", action="append", default=None)
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to benchmark (default: base).",
    )
    parser.add_argument(
        "--check-finished",
        action="store_true",
        help="report expected benchmark outputs and exit",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="delete existing benchmark outputs and rerun every step",
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
            "print every not-yet-finished step's command to stdout "
            "prefixed with `uv run` (no apptainer wrapper, unlike --local "
            "--dry-run) and exit -- for hand-picking from on a cluster "
            "with the cherimoya extra installed (`uv sync --extra "
            "cherimoya`), or inside an already-entered Apptainer shell "
            "(no SLURM submission, no job bundling); combine with --rerun "
            "to print every step regardless"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "run each variant's steps directly in the foreground, instead "
            "of submitting SLURM jobs; ignores SLURM resource flags"
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        default=default_apptainer_image(),
        help=f"path to the Cherimoya Apptainer .sif image (default: {default_apptainer_image()})",
    )
    parser.add_argument("--apptainer-bind", action="append", default=None)
    parser.add_argument("--save-cross-outputs", action="store_true")
    parser.add_argument(
        "--no-save-da-outputs",
        dest="save_da_outputs",
        action="store_false",
        help="Do not pass --save-output to benchmark_da.py.",
    )
    parser.set_defaults(save_da_outputs=True)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
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
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", type=str, default="2:00:00")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    species = species_config.resolve_species(args.species)
    choices = species_config.read_pseudobulks(species, args.dataset)
    args.pseudobulks = args.pseudobulks or choices
    for pseudobulk in args.pseudobulks:
        if pseudobulk not in choices:
            print(
                f"Error: unknown pseudobulk '{pseudobulk}' for species '{species}', "
                f"dataset '{args.dataset}'. Choose one of: {', '.join(choices)}",
                file=sys.stderr,
            )
            sys.exit(1)

    jobs = build_jobs(args, species)
    if args.check_finished:
        print_finished_report(jobs)
        return

    if args.rerun:
        for entries in jobs.values():
            for entry in entries:
                entry["finish_path"].unlink(missing_ok=True)

    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)
    apptainer_image_arg = shlex.quote(args.apptainer_image)

    if args.print_commands:
        for entries in jobs.values():
            for entry in entries:
                if entry["finish_path"].exists():
                    continue
                print(shlex.join(["uv", "run", *entry["command"]]))
        return

    log_dir = species_config.logs_dir(species) / "cherimoya_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
    skipped_empty = 0
    skipped_steps = 0
    local_failures = []

    for variant_name, entries in jobs.items():
        skipped_steps += sum(entry["finish_path"].exists() for entry in entries)
        if all(entry["finish_path"].exists() for entry in entries):
            skipped_empty += 1
            continue

        job_name = slurm_name(f"benchmark_{species}_{args.dataset}_{variant_name}")

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
                if args.dry_run:
                    print(shlex.join(full_cmd))
                    continue
                print(f"=== {job_name}: {entry['label']} ===")
                result = subprocess.run(full_cmd, cwd=REPO_ROOT)
                if result.returncode != 0:
                    local_failures.append((job_name, entry["label"], result.returncode))
                    print(
                        f"ERROR: step exited with {result.returncode}", file=sys.stderr
                    )
            submitted += 1
            continue

        # Every selected step is embedded in the script (not just the ones
        # still pending at submission time), each guarded by a runtime check
        # of its own output-file completion marker -- mirrors
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
            run_cmd = (
                f"apptainer exec --nv {apptainer_bind_args} {apptainer_image_arg} "
                + " ".join(shlex.quote(part) for part in entry["command"])
            )
            step_lines += [
                f"if [[ -f {finish_path_bash} ]]; then",
                f'    echo "Skipping already-finished step: {entry["label"]}"',
                "else",
                f'    echo "Running step: {entry["label"]}"',
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
        unit = "variant job(s)"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "sbatch job(s)"
    print(
        f"\n{action} {submitted} {unit}; skipped {skipped_empty} fully-finished "
        f"variant(s), {skipped_steps} already-finished step(s) (skipped at runtime)."
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
