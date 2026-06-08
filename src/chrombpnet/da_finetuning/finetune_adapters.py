#!/usr/bin/env python3
"""Compatibility wrapper for daicon adapter DA finetuning.

The main implementation now lives in ``finetune_da.py --mode adapter``. This
wrapper preserves the historical adapter CLI used by sweep launchers.
"""

import argparse

from finetune_da import PSEUDOBULKS, main as finetune_da_main


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--pseudobulk",
        required=True,
        choices=PSEUDOBULKS,
        help="Cell type to finetune",
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
        default=["all"],
        help="Adapter layer targets or presets (default: all)",
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=100)
    parser.add_argument("--jitter", type=int, default=200)
    parser.add_argument("--da-fraction", type=float, default=0.5)
    parser.add_argument(
        "--materialize-model",
        action="store_true",
        help="Also save a full .torch model with adapters injected",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _main_argv(args):
    argv = [
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
        "--alpha",
        str(args.alpha),
        "--jitter",
        str(args.jitter),
        "--da-fraction",
        str(args.da_fraction),
    ]
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
