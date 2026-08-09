#!/usr/bin/env python3
"""Train a Cherimoya accessibility model for one pseudobulk/fold.

cherimoya is a Linux-only core dependency of this project (its torch>=2.9/
triton>=3.5.1 requirement has no macOS wheels) -- run this with `uv run` on
a Linux GPU machine.

Train/validation chromosomes come from the existing
configs/<species>/fold_N.json files (unchanged from the ChromBPNet
pipeline): fold_N["train"] / fold_N["valid"] are passed to Cherimoya;
fold_N["test"] is never touched here -- held out entirely for benchmark.py.

Negatives are GC-matched regions pre-generated per pseudobulk by
negatives.py (Cherimoya's own `negatives` CLI, not fold-specific).

This resolves our paths/hyperparameters into a JSON parameters file and
shells out to `cherimoya fit -p <file>` directly, rather than reimplementing
its optimizer/scheduler setup (Muon for 2D projection weights, AdamW for
everything else, SGD for the lw0/lw1 Kendall loss weights, warmup+cosine
schedules) in Python -- that split is exactly what the CLI already does, so
hand-rolling it here only risked drifting from Cherimoya's own recommended
defaults. `cherimoya fit` also runs its own evaluate step on
`validation_chroms` afterward, writing `{name}.evaluate.json`/
`{name}.performance.tsv` alongside the model as a side effect; that's a
validation-set sanity check, not a substitute for benchmark.py's held-out
test-chromosome evaluation. The generated parameters JSON is kept next to
the model for reproducibility.

Usage:
    uv run python src/cherimoya_pipeline/fit/train_cherimoya.py -p bundle_sheath -f 0
    uv run python src/cherimoya_pipeline/fit/train_cherimoya.py --species A.thaliana -p mesophyll -f 2
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

DATASETS = ("base", "fine", "coarse")
IN_WINDOW = 2114
OUT_WINDOW = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument("-p", "--pseudobulk", required=True)
    parser.add_argument(
        "-f", "--fold", type=int, required=True, choices=range(5), help="Fold index"
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to train (default: base).",
    )
    parser.add_argument("-o", "--output-dir", type=str, default=None)
    parser.add_argument("--n-filters", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--max-jitter", type=int, default=500)
    parser.add_argument(
        "--n-warmup-epochs",
        type=int,
        default=2,
        help="Warmup epochs for the LR schedules (default: 2, cherimoya's own default).",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help="Negatives per positive, per batch (default: 0.25).",
    )
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument(
        "--blacklist",
        type=str,
        default=None,
        help="Optional BED of regions to exclude.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    species = species_config.resolve_species(args.species)
    pseudobulks = species_config.read_pseudobulks(species, args.dataset)
    if args.pseudobulk not in pseudobulks:
        print(
            f"Error: unknown pseudobulk '{args.pseudobulk}' for species '{species}', "
            f"dataset '{args.dataset}'. Choose one of: {', '.join(pseudobulks)}",
            file=sys.stderr,
        )
        sys.exit(1)

    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    data_dir = species_config.data_dir(species)
    model_root = species_config.models_dir(species)
    if args.dataset != "base":
        model_root = model_root / args.dataset

    fasta = str(species_config.genome_fasta(species))
    peaks_path = str(
        data_dir / f"peaks{dataset_suffix}" / f"{args.pseudobulk}_peaks.narrowPeak"
    )
    bw_path = str(data_dir / f"bigwigs{dataset_suffix}" / f"{args.pseudobulk}.bw")
    negatives_path = str(
        data_dir / f"negatives{dataset_suffix}" / f"{args.pseudobulk}_negatives.bed"
    )
    for path, label in [
        (peaks_path, "peaks"),
        (bw_path, "bigwig"),
        (negatives_path, "negatives"),
    ]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    with open(species_config.fold_path(species, args.fold)) as f:
        fold = json.load(f)
    train_chroms, valid_chroms = fold["train"], fold["valid"]
    print(f"Fold {args.fold}: train={train_chroms}, valid={valid_chroms}")

    output_dir = Path(args.output_dir or str(model_root / args.pseudobulk))
    output_dir.mkdir(parents=True, exist_ok=True)
    name = str(output_dir / f"{args.pseudobulk}_f{args.fold}")

    exclusion_lists = [args.blacklist] if args.blacklist else None

    # Keys/shape follow cherimoya_cli.defaults.default_fit_parameters; any
    # key we omit falls back to that CLI's own default via its
    # merge_parameters(). signal_groups/n_control_tracks/trimming aren't
    # set here -- `cherimoya fit` derives them itself from `signals`/
    # `controls`/`in_window`/`out_window`.
    parameters = {
        "n_filters": args.n_filters,
        "n_layers": args.n_layers,
        "name": name,
        "batch_size": args.batch_size,
        "in_window": IN_WINDOW,
        "out_window": OUT_WINDOW,
        "max_jitter": args.max_jitter,
        "reverse_complement": True,
        "max_epochs": args.max_epochs,
        "n_warmup_epochs": args.n_warmup_epochs,
        "negative_ratio": args.negative_ratio,
        "num_workers": 0,
        "dtype": "float32",
        "device": "cuda",
        "early_stopping": args.early_stopping,
        "verbose": args.verbose,
        "training_chroms": train_chroms,
        "validation_chroms": valid_chroms,
        "sequences": fasta,
        "loci": peaks_path,
        "exclusion_lists": exclusion_lists,
        "negatives": negatives_path,
        "signals": [bw_path],
        "controls": None,
        "random_state": args.random_state,
    }

    params_path = Path(f"{name}.fit_params.json")
    with open(params_path, "w") as f:
        json.dump(parameters, f, indent=4)
    print(f"Wrote {params_path}")

    subprocess.run(["cherimoya", "fit", "-p", str(params_path)], check=True)
    print(f"\nModel saved to {name}.torch")


if __name__ == "__main__":
    main()
