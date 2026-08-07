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

Usage:
    uv run python src/cherimoya/fit/train_cherimoya.py -p bundle_sheath -f 0
    uv run python src/cherimoya/fit/train_cherimoya.py --species A.thaliana -p mesophyll -f 2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from cherimoya import Cherimoya
from cherimoya.io import PeakGenerator
from tangermeme.io import extract_loci
from torch.optim import SGD, AdamW, Muon
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

torch.backends.cudnn.benchmark = True

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

    blacklist = [args.blacklist] if args.blacklist else None

    print(f"Loading training data (chroms: {train_chroms})...")
    train_data_loader = PeakGenerator(
        peaks=peaks_path,
        negatives=negatives_path,
        sequences=fasta,
        signals=[bw_path],
        chroms=train_chroms,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        max_jitter=args.max_jitter,
        reverse_complement=True,
        shuffle=True,
        random_state=args.random_state,
        batch_size=args.batch_size,
        verbose=args.verbose,
        negative_ratio=args.negative_ratio,
        exclusion_lists=blacklist,
        pin_memory=True,
        num_workers=0,
    )

    print(f"Loading validation data (chroms: {valid_chroms})...")
    X_valid, y_valid = extract_loci(
        loci=peaks_path,
        sequences=fasta,
        signals=[bw_path],
        chroms=valid_chroms,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        max_jitter=0,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        exclusion_lists=blacklist,
        verbose=args.verbose,
    )
    y_valid = torch.abs(y_valid)

    # torch.compile() unconditionally raises on Python 3.14+ before torch
    # 2.10 (https://github.com/pytorch/pytorch/issues/169875); disable it
    # automatically on that combination rather than crashing.
    compile_supported = sys.version_info < (3, 14) or torch.__version__ >= "2.10"
    model = Cherimoya(
        name=name,
        n_filters=args.n_filters,
        signal_groups=[1],
        n_control_tracks=0,
        n_layers=args.n_layers,
        trimming=(IN_WINDOW - OUT_WINDOW) // 2,
        verbose=args.verbose,
        compile=compile_supported,
    )
    model = model.to("cuda")

    # Muon for 2D projection weights, AdamW for everything else, plain SGD
    # for the lw0/lw1 Kendall uncertainty loss weights -- matches Cherimoya's
    # own recommended optimizer split (cherimoya_cli fit command).
    muon_params, adam_params, lw_params = [], [], []
    for pname, p in model.named_parameters():
        if pname in ("lw0", "lw1"):
            lw_params.append(p)
        elif (
            p.ndim == 2
            and "weight" in pname
            and pname != "linear.weight"
            and "conv_weight" not in pname
        ):
            muon_params.append(p)
        else:
            adam_params.append(p)

    muon_optimizer = Muon(muon_params, lr=0.025, weight_decay=0.03)
    adam_optimizer = AdamW(adam_params, lr=0.001, weight_decay=0.0)
    lw_optimizer = SGD(lw_params, lr=0.001, weight_decay=0.0, momentum=0.9)

    num_warmup_epochs = min(5, args.max_epochs)
    num_warmup_iters = len(train_data_loader) * num_warmup_epochs
    num_decay_iters = len(train_data_loader) * max(
        1, args.max_epochs - num_warmup_epochs
    )

    def warmup_then_cosine(optimizer):
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_iters),
                CosineAnnealingLR(optimizer, T_max=num_decay_iters, eta_min=1e-5),
            ],
            milestones=[num_warmup_iters],
        )

    muon_scheduler = warmup_then_cosine(muon_optimizer)
    adam_scheduler = warmup_then_cosine(adam_optimizer)
    # Linear warmup then flat (no decay) for the Kendall loss weights.
    lw_scheduler = SequentialLR(
        lw_optimizer,
        schedulers=[
            LinearLR(lw_optimizer, start_factor=0.01, total_iters=num_warmup_iters),
            ConstantLR(lw_optimizer, factor=1.0, total_iters=1),
        ],
        milestones=[num_warmup_iters],
    )

    model.fit(
        training_data=train_data_loader,
        muon_optimizer=muon_optimizer,
        adam_optimizer=adam_optimizer,
        lw_optimizer=lw_optimizer,
        muon_scheduler=muon_scheduler,
        adam_scheduler=adam_scheduler,
        lw_scheduler=lw_scheduler,
        X_valid=X_valid,
        X_ctl_valid=None,
        y_valid=y_valid,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        dtype=torch.float32,
    )

    print(f"\nModel saved to {name}.torch")


if __name__ == "__main__":
    main()
