"""
Benchmark a trained ChromBPNet model (ported to torch with bpnetlite) across all folds.

Loads trained models for each fold, predicts on the held-out test chromosomes,
and reports profile and count prediction metrics.

Usage:
    uv run python src/chrombpnet/benchmark/benchmark.py -p mesophyll
    uv run python src/chrombpnet/benchmark/benchmark.py -p mesophyll --finetuned
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from bpnetlite.performance import jensen_shannon_distance, pearson_corr, spearman_corr
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from daicon import load_model_with_adapter  # noqa: E402
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402

CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
DATASETS = ("base", "fine", "coarse")


def read_pseudobulks(dataset):
    suffix = "" if dataset == "base" else f"_{dataset}"
    path = REPO_ROOT / "configs" / f"pseudobulks{suffix}.txt"
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--pseudobulk",
        type=str,
        required=True,
        help="The name of the pseudobulk to benchmark",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to benchmark (default: base).",
    )
    parser.add_argument(
        "-o",
        "--save-output",
        action="store_true",
        help="save predictions and signals to disk",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned models instead of base torch/ models",
        adapter_help="Use default daicon adapter bundle for the selected DA fraction",
    )
    args = parser.parse_args()

    variant = resolve_model_variant(args)
    if args.pseudobulk not in read_pseudobulks(args.dataset):
        choices = ", ".join(read_pseudobulks(args.dataset))
        print(
            f"Error: unknown pseudobulk '{args.pseudobulk}' for dataset '{args.dataset}'. "
            f"Choose one of: {choices}",
            file=sys.stderr,
        )
        sys.exit(1)
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]
    metrics_subdir = variant["metrics_subdir"]
    predictions_subdir = variant["predictions_subdir"]
    dataset_suffix = "" if args.dataset == "base" else f"_{args.dataset}"
    model_root = REPO_ROOT / "models"
    if args.dataset != "base":
        model_root = model_root / args.dataset

    # Load chrom splits
    with open(CHROM_SPLITS_PATH, "r") as f:
        chrom_splits = json.load(f)
    n_folds = len(chrom_splits)

    # Construct data paths and verify that they exist.
    peaks_path = str(
        REPO_ROOT
        / "data"
        / f"peaks{dataset_suffix}"
        / f"{args.pseudobulk}_peaks.narrowPeak"
    )
    bw_path = str(
        REPO_ROOT / "data" / f"bigwigs{dataset_suffix}" / f"{args.pseudobulk}.bw"
    )
    for path, label in [(peaks_path, "peaks"), (bw_path, "bigwig")]:
        if not Path(path).exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Construct model paths and verify that they exist.
    model_paths = [
        model_root / args.pseudobulk / model_subdir / f"{args.pseudobulk}_f{fold}.torch"
        for fold in range(n_folds)
    ]
    adapter_paths = (
        [
            model_root
            / args.pseudobulk
            / "adapters"
            / adapter_id
            / f"{args.pseudobulk}_f{fold}.adapter.pt"
            for fold in range(n_folds)
        ]
        if adapter_id is not None
        else [None] * n_folds
    )
    for model_path in model_paths:
        if not model_path.exists():
            print(f"Error: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)
    for adapter_path in adapter_paths:
        if adapter_path is not None and not adapter_path.exists():
            print(f"Error: adapter bundle not found: {adapter_path}", file=sys.stderr)
            sys.exit(1)

    params = {
        "sequences": FASTA,
        "signals": [bw_path],
        "loci": peaks_path,
        "in_window": 2114,
        "out_window": 1000,
        "batch_size": 64,
        "verbose": False,
    }
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.verbose:
        params["verbose"] = True

    # Predict on each fold's test chromosomes
    signals = []
    preds = []
    for fold in range(n_folds):
        test_chroms = chrom_splits[str(fold)]
        X, y = extract_loci(
            loci=params["loci"],
            sequences=params["sequences"],
            chroms=test_chroms,
            signals=params["signals"],
            in_window=params["in_window"],
            out_window=params["out_window"],
            verbose=params["verbose"],
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        )
        signals.append(torch.abs(y))

        model, _manifest = load_model_with_adapter(
            model_paths[fold],
            adapter_paths[fold],
            map_location=torch.device("cpu"),
            adapter_root="accessibility",
        )
        preds.append(
            predict(
                model=model,
                X=X,
                verbose=params["verbose"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=params["batch_size"],
            )
        )

    # Calculate performance metrics
    profile_corr = [
        pearson_corr(
            torch.nn.functional.softmax(pred[0].reshape(pred[0].shape[0], -1), dim=-1),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]
    profile_jsd = [
        jensen_shannon_distance(
            torch.nn.functional.log_softmax(
                pred[0].reshape(pred[0].shape[0], -1), dim=-1
            ),
            signal.reshape(pred[0].shape[0], -1),
        ).numpy()
        for pred, signal in zip(preds, signals)
    ]
    log_counts_pearson = [
        pearson_corr(pred[1].squeeze(), torch.log1p(signal.sum(dim=(-1, -2)))).item()
        for pred, signal in zip(preds, signals)
    ]
    counts_spearman = [
        spearman_corr(pred[1].squeeze(), signal.sum(dim=(-1, -2))).item()
        for pred, signal in zip(preds, signals)
    ]
    log_counts_pearson_all = pearson_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([torch.log1p(signal.sum(dim=(-1, -2))) for signal in signals]),
    ).item()
    counts_spearman_all = spearman_corr(
        torch.cat([pred[1].squeeze() for pred in preds]),
        torch.cat([signal.sum(dim=(-1, -2)) for signal in signals]),
    ).item()

    print("\nPer-fold results:\n----------------")
    print(
        f"Profile Pearson correlation: {[np.nanmedian(c).item() for c in profile_corr]}"
        f" (n_nan={[np.isnan(c).mean().item() for c in profile_corr]})"
    )
    print(
        f"Profile Jensen-Shannon distance: {[np.nanmedian(j).item() for j in profile_jsd]} "
        f"(n_nan={[np.isnan(j).mean().item() for j in profile_jsd]})"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson}")
    print(f"Counts Spearman correlation: {counts_spearman}")

    print("\nGenome-wide results:\n----------------")
    print(f"Profile Pearson correlation: {np.nanmedian(np.concatenate(profile_corr))}")
    print(
        f"Profile Jensen-Shannon distance: {np.nanmedian(np.concatenate(profile_jsd))}"
    )
    print(f"Log counts Pearson correlation: {log_counts_pearson_all}")
    print(f"Counts Spearman correlation: {counts_spearman_all}")

    # Save metrics to JSON
    metrics = {
        "pseudobulk": args.pseudobulk,
        "dataset": args.dataset,
        "per_fold": {
            str(fold): {
                "profile_pearson": np.nanmedian(profile_corr[fold]).item(),
                "profile_jsd": np.nanmedian(profile_jsd[fold]).item(),
                "log_counts_pearson": log_counts_pearson[fold],
                "counts_spearman": counts_spearman[fold],
            }
            for fold in range(n_folds)
        },
        "genome_wide": {
            "profile_pearson": np.nanmedian(np.concatenate(profile_corr)).item(),
            "profile_jsd": np.nanmedian(np.concatenate(profile_jsd)).item(),
            "log_counts_pearson": log_counts_pearson_all,
            "counts_spearman": counts_spearman_all,
        },
    }
    metrics_dir = (
        REPO_ROOT
        / "performance_metrics"
        / (metrics_subdir if args.dataset == "base" else f"{metrics_subdir}_{args.dataset}")
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{args.pseudobulk}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    if args.save_output:
        output_dir = (
            REPO_ROOT
            / "predictions"
            / (
                predictions_subdir
                if args.dataset == "base"
                else f"{predictions_subdir}_{args.dataset}"
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.pseudobulk}.npz"
        scaled_preds = {
            f"predict_f{i}": (
                torch.nn.functional.softmax(
                    pred[0].reshape(pred[0].shape[0], -1), dim=-1
                )
                * torch.exp(pred[1])
            )
            .reshape(*pred[0].shape)
            .numpy()
            for i, pred in enumerate(preds)
        }
        expts = {f"expt_f{i}": signal.numpy() for i, signal in enumerate(signals)}
        np.savez_compressed(output_path, **scaled_preds, **expts)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
