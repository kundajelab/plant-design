#!/usr/bin/env python3
"""Collect and print unfinished Cherimoya benchmark jobs for active model variants.

This launcher covers the main reporting path:

- base Cherimoya
- full-model DA fine-tuning at 0.5 and 0.8 DA fractions
- daicon adapter DA fine-tuning at 0.5 and 0.8 DA fractions

Development-only adapter parameter sweeps are handled separately by
``benchmark_lora_sweep.py``.

Prints one shell command per missing benchmark output; pipe into `parallel`
or run directly. Jobs run via `uv run --frozen` inside the shared
plant-design environment.
"""

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

BENCHMARK_DIR = REPO_ROOT / "src" / "cherimoya_pipeline" / "benchmark"
DATASETS = ("base", "fine", "coarse")
PYTHON = ("uv", "run", "--frozen", "python")


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


def build_variant_job(args, species, variant):
    commands = []
    skipped = 0

    for pseudobulk in args.pseudobulks:
        if (
            existing_cross_path(species, args.dataset, variant, pseudobulk) is not None
            and not args.rerun
        ):
            skipped += 1
            continue
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark.py",
            "--species",
            species,
            "--dataset",
            args.dataset,
            "-p",
            pseudobulk,
            "-b",
            args.batch_size,
            *variant.flags,
        ]
        if args.save_cross_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        commands.append(" ".join(shlex.quote(str(part)) for part in cmd))

    da_path = existing_da_path(species, args.dataset, variant)
    da_done = da_path is not None and (
        not args.save_da_outputs or benchmark_da_peaks_path(da_path).exists()
    )
    if not da_done or args.rerun:
        cmd = [
            *PYTHON,
            BENCHMARK_DIR / "benchmark_da.py",
            "--species",
            species,
            "--dataset",
            args.dataset,
            "-b",
            args.batch_size,
            *variant.flags,
        ]
        if args.save_da_outputs:
            cmd.append("--save-output")
        if args.verbose:
            cmd.append("-v")
        commands.append(" ".join(shlex.quote(str(part)) for part in cmd))
    else:
        skipped += 1

    if args.dataset != "base":
        return commands, skipped

    pairwise_path = species_config.performance_metrics_dir(species) / variant.pairwise_path
    if not pairwise_path.exists() or args.rerun:
        commands.append(
            " ".join(
                shlex.quote(str(part))
                for part in [
                    *PYTHON,
                    BENCHMARK_DIR / "pairwise_correlations_predicted.py",
                    "--species",
                    species,
                    *variant.flags,
                ]
            )
        )
    else:
        skipped += 1

    return commands, skipped


def print_jobs(jobs):
    printed = 0
    skipped = 0
    for commands, n_skipped in jobs:
        for command in commands:
            print(command)
            printed += 1
        skipped += n_skipped
    print(
        f"\nPrinted {printed} command(s); skipped {skipped} already-complete step(s).",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        default=None,
        help="Model variant to benchmark; repeatable. Default: all active variants.",
    )
    parser.add_argument(
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to benchmark (default: base).",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--save-cross-outputs", action="store_true")
    parser.add_argument(
        "--no-save-da-outputs",
        dest="save_da_outputs",
        action="store_false",
        help="Do not pass --save-output to benchmark_da.py.",
    )
    parser.set_defaults(save_da_outputs=True)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
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
    variant_names = args.variant or list(VARIANTS)
    jobs = [build_variant_job(args, species, VARIANTS[name]) for name in variant_names]
    print_jobs(jobs)


if __name__ == "__main__":
    main()
