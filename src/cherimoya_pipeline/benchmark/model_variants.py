"""Shared model-variant CLI helpers for benchmark scripts."""

import sys

from cherimoya import Cherimoya
from daicon import load_adapter_bundle


def load_cherimoya_with_adapter(model_path, adapter_path=None, device="cpu"):
    """Load a Cherimoya checkpoint, optionally injecting a daicon adapter.

    Cherimoya checkpoints are a config + state_dict bundle (loaded via
    Cherimoya.load), not a pickled nn.Module like ChromBPNet's -- so this
    does not go through daicon.load_model_with_adapter, which assumes the
    latter. Adapters are applied at the model root: Cherimoya has no
    bias/accessibility submodel split to target.
    """
    model = Cherimoya.load(model_path, device=device, compile=False)
    if adapter_path is None:
        return model, None
    return load_adapter_bundle(
        model, adapter_path, map_location=device, adapter_root=None
    )


def finetuned_tag(da_fraction):
    return "finetuned" if da_fraction == 0.5 else f"finetuned_{da_fraction:g}"


def finetuned_metrics_subdir(da_fraction):
    return f"cherimoya_{finetuned_tag(da_fraction)}"


def daicon_adapter_id(da_fraction):
    da_tag = str(da_fraction).replace(".", "p")
    return f"daicon_default_da{da_tag}"


def add_model_variant_args(parser, *, finetuned_help, adapter_help):
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help=finetuned_help,
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects model/output subdirectory (default: 0.5)",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help=adapter_help,
    )
    parser.add_argument(
        "--adapter-sweep-id",
        "--adapter-run-id",
        dest="adapter_sweep_id",
        type=str,
        default=None,
        help="Use adapter bundle from models/{pseudobulk}/adapters/{adapter_id}/",
    )


def resolve_model_variant(args):
    selected = sum([args.finetuned, args.adapter, args.adapter_sweep_id is not None])
    if selected > 1:
        print(
            "Error: use only one of --finetuned, --adapter, or --adapter-sweep-id.",
            file=sys.stderr,
        )
        sys.exit(1)

    fine_tag = finetuned_tag(args.da_fraction)
    adapter_id = None
    if args.adapter:
        adapter_id = daicon_adapter_id(args.da_fraction)
    elif args.adapter_sweep_id is not None:
        adapter_id = args.adapter_sweep_id

    # Base (non-finetuned) checkpoints are written directly to
    # models/<species>/<pseudobulk>/ by train_cherimoya.py -- no subdir.
    model_subdir = fine_tag if args.finetuned else ""
    if adapter_id is not None:
        suffix = f"_adapter_{adapter_id}"
        metrics_subdir = f"adapters/{adapter_id}"
        predictions_subdir = f"adapters/{adapter_id}"
        result_tag = adapter_id
    elif args.finetuned:
        suffix = f"_{fine_tag}"
        metrics_subdir = finetuned_metrics_subdir(args.da_fraction)
        predictions_subdir = fine_tag
        result_tag = fine_tag
    else:
        suffix = ""
        metrics_subdir = "cherimoya"
        predictions_subdir = "cherimoya"
        result_tag = "cherimoya"

    return {
        "fine_tag": fine_tag,
        "adapter_id": adapter_id,
        "model_subdir": model_subdir,
        "suffix": suffix,
        "metrics_subdir": metrics_subdir,
        "predictions_subdir": predictions_subdir,
        "result_tag": result_tag,
    }
