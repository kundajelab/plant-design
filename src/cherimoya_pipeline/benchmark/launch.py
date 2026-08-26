#!/usr/bin/env python3
"""Submit SLURM jobs to benchmark active Cherimoya model variants.

This launcher covers the main reporting path:

- base Cherimoya
- full-model DA fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter DA fine-tuning at 0.5 and 0.8 DA fractions

Development-only adapter parameter sweeps are handled separately by
``benchmark_lora_sweep.py``.

One job is submitted per variant, bundling its cross-pseudobulk
benchmark.py calls plus benchmark_da.py and (base dataset only)
pairwise_correlations_predicted.py into a single sbatch script -- these
steps are individually quick, so one job per pseudobulk would mostly be
SLURM scheduling overhead.

By default, jobs run via `apptainer exec --nv` against the Cherimoya
Apptainer image (see ../apptainer/).

--local skips SLURM entirely and runs each variant's steps directly in the
foreground with inherited stdout/stderr, instead of submitting SLURM jobs.
Combined with --dry-run, --local prints one runnable command per step
instead of running them -- pipe that into something like
`simple_gpu_scheduler`.

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


def existing_cross_path(species, dataset_name, variant, pseudobulk):
    metrics_root = species_config.performance_metrics_dir(species)
    metrics_subdir = (
        variant.cross_dir
        if dataset_name == "base"
        else Path(f"{variant.cross_dir}_{dataset_name}")
    )
    path = metrics_root / metrics_subdir / f"{pseudobulk}.json"
    return path if path.exists() else None


def dataset_suffix(dataset_name):
    return "" if dataset_name == "base" else f"_{dataset_name}"


def existing_da_path(species, dataset_name, variant):
    metrics_root = species_config.performance_metrics_dir(species)
    if dataset_name == "base":
        path = metrics_root / variant.da_path
    else:
        path = (
            metrics_root
            / f"differential_accessibility{dataset_suffix(dataset_name)}"
            / variant.da_path.name
        )
    return path if path.exists() else None


def benchmark_da_peaks_path(da_path):
    # benchmark_da.py's --save-output always writes this as a .csv
    # regardless of da_path's own (.json) extension.
    return da_path.with_name(
        da_path.stem.replace("da_benchmark", "da_benchmark_peaks") + ".csv"
    )


def build_variant_steps(args, species, variant):
    """Return (list of command-part lists, skipped count) for one variant."""
    steps = []
    skipped = 0

    for pseudobulk in args.pseudobulks:
        if (
            existing_cross_path(species, args.dataset, variant, pseudobulk) is not None
            and not args.rerun
        ):
            skipped += 1
            continue
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
        steps.append(cmd)

    da_path = existing_da_path(species, args.dataset, variant)
    da_done = da_path is not None and (
        not args.save_da_outputs or benchmark_da_peaks_path(da_path).exists()
    )
    if not da_done or args.rerun:
        cmd = [
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
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        steps.append(cmd)
    else:
        skipped += 1

    if args.dataset != "base":
        return steps, skipped

    pairwise_path = species_config.performance_metrics_dir(species) / variant.pairwise_path
    if not pairwise_path.exists() or args.rerun:
        steps.append(
            [
                "python",
                str(BENCHMARK_DIR / "pairwise_correlations_predicted.py"),
                "--species",
                species,
                *variant.flags,
            ]
        )
    else:
        skipped += 1

    return steps, skipped


def apptainer_wrap(cmd, apptainer_image, apptainer_bind_args_list):
    return [
        "apptainer",
        "exec",
        "--nv",
        *[part for path in apptainer_bind_args_list for part in ("--bind", path)],
        apptainer_image,
        *cmd,
    ]


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
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print sbatch scripts (or local commands, with --local) without running them",
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

    variant_names = args.variant or (["base"] if args.dataset != "base" else list(VARIANTS))
    log_dir = species_config.logs_dir(species) / "cherimoya_benchmark"
    log_dir.mkdir(parents=True, exist_ok=True)
    apptainer_binds = args.apptainer_bind or [str(REPO_ROOT)]
    apptainer_bind_args = " ".join(f"--bind {shlex.quote(p)}" for p in apptainer_binds)

    submitted = 0
    skipped_empty = 0
    skipped_steps = 0
    local_failures = []

    for name in variant_names:
        variant = VARIANTS[name]
        steps, n_skipped = build_variant_steps(args, species, variant)
        skipped_steps += n_skipped
        if not steps:
            skipped_empty += 1
            continue

        job_name = slurm_name(f"benchmark_{args.dataset}_{variant.name}")

        if args.local:
            for cmd in steps:
                full_cmd = apptainer_wrap(cmd, args.apptainer_image, apptainer_binds)
                if args.dry_run:
                    print(shlex.join(full_cmd))
                    continue
                print(f"=== {job_name}: {' '.join(cmd)} ===")
                result = subprocess.run(full_cmd, cwd=REPO_ROOT)
                if result.returncode != 0:
                    local_failures.append((job_name, cmd, result.returncode))
                    print(f"ERROR: step exited with {result.returncode}", file=sys.stderr)
            submitted += 1
            continue

        # Built with explicit indentation (rather than shlex.join()-ing the
        # whole line, which would quote the "$APPTAINER_IMAGE" placeholder
        # itself into '$APPTAINER_IMAGE' and break bash variable expansion)
        # and joined with the same 12-space indent as the rest of the
        # template below, so the final textwrap.dedent() still finds a
        # common prefix across every line -- a mismatched indent here
        # leaves the whole script indented, which sbatch rejects as "not a
        # batch script".
        step_lines = [
            f'time apptainer exec --nv {apptainer_bind_args} "$APPTAINER_IMAGE" '
            + " ".join(shlex.quote(part) for part in cmd)
            for cmd in steps
        ]
        run_block = "\n\n            ".join(step_lines)
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

            {run_block}
        """)

        if args.dry_run:
            print(f"--- {job_name} ({len(steps)} step(s), skipped {n_skipped}) ---")
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
        unit = "variant(s)"
    else:
        action = "Would submit" if args.dry_run else "Submitted"
        unit = "sbatch job(s)"
    print(
        f"\n{action} {submitted} {unit}; skipped {skipped_empty} empty variant(s), "
        f"{skipped_steps} already-complete step(s)."
    )
    if args.local and not args.dry_run:
        if local_failures:
            print(f"{len(local_failures)} step(s) failed:")
            for job_name, cmd, returncode in local_failures:
                print(f"  {job_name}: {' '.join(cmd)} (exit {returncode})")
        else:
            print("All steps completed successfully.")


if __name__ == "__main__":
    main()
