#!/usr/bin/env python3
"""Compatibility wrapper for daicon adapter DA finetuning.

The main implementation now lives in ``finetune_da.py --mode adapter``. This
wrapper preserves the historical adapter CLI used by sweep launchers.
"""

import argparse
import sys
from pathlib import Path

from finetune_da import DATASETS, main as finetune_da_main

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "-p",
        "--pseudobulk",
        required=True,
        help="Cell type to finetune",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to finetune (default: base).",
    )
    parser.add_argument("-f", "--fold", type=int, required=True, help="Fold index")
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="Name for this adapter run under models/{pseudobulk}/adapters/",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all_postblock"],
        help="Adapter layer targets or presets (default: all_postblock)",
    )
    parser.add_argument("--conv-rank", type=int, default=16)
    parser.add_argument("--linear-rank", type=int, default=4)
    parser.add_argument(
        "--conv-alpha",
        type=float,
        default=None,
        help="Legacy conv adapter alpha; translated to --conv-scale alpha / conv_rank",
    )
    parser.add_argument(
        "--linear-alpha",
        type=float,
        default=None,
        help="Legacy linear adapter alpha; translated to --linear-scale alpha / linear_rank",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-jitter", type=int, default=500)
    parser.add_argument("--da-fraction", type=float, default=0.5)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument(
        "--blacklist",
        type=str,
        default=None,
        help="Optional BED of regions to exclude",
    )
    parser.add_argument(
        "--materialize-model",
        action="store_true",
        help="Also save a full .torch model with adapters injected",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _main_argv(args):
    argv = [
        "--species",
        species_config.resolve_species(args.species),
        "--dataset",
        args.dataset,
        "-p",
        args.pseudobulk,
        "-f",
        str(args.fold),
        "--mode",
        "adapter",
        "--adapter-run-id",
        args.sweep_id,
        "--adapter-targets",
        *args.targets,
        "--conv-rank",
        str(args.conv_rank),
        "--linear-rank",
        str(args.linear_rank),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--max-jitter",
        str(args.max_jitter),
        "--da-fraction",
        str(args.da_fraction),
        "--negative-ratio",
        str(args.negative_ratio),
    ]
    if args.blacklist is not None:
        argv.extend(["--blacklist", args.blacklist])
    if args.conv_alpha is not None:
        argv.extend(["--conv-scale", str(args.conv_alpha / args.conv_rank)])
    if args.linear_alpha is not None:
        argv.extend(["--linear-scale", str(args.linear_alpha / args.linear_rank)])
    if args.materialize_model:
        argv.append("--materialize-model")
    if args.verbose:
        argv.append("--verbose")
    return argv


def main(argv=None):
    args = build_parser().parse_args(argv)
    finetune_da_main(_main_argv(args))


if __name__ == "__main__":
    main()
