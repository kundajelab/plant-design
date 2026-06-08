#!/usr/bin/env python3
"""Development-only summary plots for ChromBPNet low-rank adapter sweeps.

Reads benchmark outputs produced by benchmark_lora_sweep.py and produces a
summary CSV plus compact PDF heatmaps. By default, summaries include the base
ChromBPNet and full-model finetuned references.

Use plot_benchmark.py for the active benchmark path.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "src" / "chrombpnet" / "benchmark"
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
REFERENCE_MODELS = {
    "base": {
        "label": "Base",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted.json",
        "benchmark_flags": [],
    },
    "finetuned": {
        "label": "Finetuned 50:50",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark_finetuned.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned.json",
        "benchmark_flags": ["--finetuned"],
    },
    "finetuned_0.8": {
        "label": "Finetuned 80:20",
        "cross_dir": REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned_0.8",
        "da_path": REPO_ROOT / "performance_metrics" / "differential_accessibility" / "da_benchmark_finetuned_0.8.json",
        "pairwise_path": REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted_finetuned_0.8.json",
        "benchmark_flags": ["--finetuned", "--da-fraction", "0.8"],
    },
}


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def finite_mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def discover_sweep_ids():
    sweep_ids = set()
    for path in (REPO_ROOT / "models").glob("*/adapters/*"):
        if path.is_dir():
            sweep_ids.add(path.name)
    for path in (REPO_ROOT / "performance_metrics" / "adapters").glob("*"):
        if path.is_dir():
            sweep_ids.add(path.name)
    return sorted(sweep_ids)


def _row(model_id, model_type, label, pseudobulk, benchmark, metric, value, sd=np.nan, n=np.nan):
    return {
        "benchmark": benchmark,
        "model_id": model_id,
        "model_type": model_type,
        "label": label,
        "pseudobulk": pseudobulk,
        "metric": metric,
        "value": value,
        "sd": sd,
        "n": n,
    }


def collect_cross_loci_rows(sweep_ids, pseudobulks, include_references=True):
    rows = []
    metrics = [
        "profile_pearson",
        "profile_jsd",
        "log_counts_pearson",
        "counts_spearman",
    ]
    sources = []
    if include_references:
        sources.extend(
            (
                model_id,
                "reference",
                spec["label"],
                spec["cross_dir"],
            )
            for model_id, spec in REFERENCE_MODELS.items()
        )
    sources.extend(
        (
            sweep_id,
            "lora_adapter",
            sweep_id,
            REPO_ROOT / "performance_metrics" / "adapters" / sweep_id,
        )
        for sweep_id in sweep_ids
    )

    for model_id, model_type, label, metrics_dir in sources:
        for pseudobulk in pseudobulks:
            path = metrics_dir / f"{pseudobulk}.json"
            data = load_json(path)
            if data is None:
                continue
            for metric in metrics:
                fold_values = [
                    data["per_fold"][fold][metric]
                    for fold in sorted(data["per_fold"], key=int)
                ]
                rows.append(
                    _row(
                        model_id,
                        model_type,
                        label,
                        pseudobulk,
                        "cross_loci",
                        metric,
                        float(np.mean(fold_values)),
                        float(np.std(fold_values, ddof=1)),
                        len(fold_values),
                    )
                )
    return rows


def _collect_one_da(model_id, model_type, label, path):
    data = load_json(path)
    if data is None:
        return []

    log2fc = data.get("log2fc_per_celltype") or {}
    return [
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "mean_log2fc_pearson",
            finite_mean([log2fc.get(ps, {}).get("pearson") for ps in PSEUDOBULKS]),
            n=len(PSEUDOBULKS),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "mean_log2fc_pearson_tau_filtered",
            finite_mean([log2fc.get(ps, {}).get("pearson_tau_filtered") for ps in PSEUDOBULKS]),
            n=len(PSEUDOBULKS),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "mean_pairwise_pearson_filtered",
            finite_mean(
                [m.get("pearson_pairwise_filtered") for m in data.get("pairwise_log2fc", {}).values()]
            ),
            n=len(data.get("pairwise_log2fc", {})),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "mean_pairwise_direction_balanced_accuracy",
            finite_mean(
                [
                    m.get("balanced_accuracy")
                    for m in data.get("pairwise_direction_concordance", {}).values()
                ]
            ),
            n=len(data.get("pairwise_direction_concordance", {})),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "tau_auprc_all_peaks",
            data.get("tau_auc", {}).get("auprc_all_peaks", np.nan),
            n=data.get("n_peaks", np.nan),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "specificity_accuracy_pairwise_filtered",
            data.get("specificity_accuracy", {}).get("pairwise_filtered", np.nan),
            n=data.get("specificity_accuracy", {}).get("n_pairwise_filtered", np.nan),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "top2_specificity_accuracy_pairwise_filtered",
            data.get("top2_specificity_accuracy", {}).get("pairwise_filtered", np.nan),
            n=data.get("specificity_accuracy", {}).get("n_pairwise_filtered", np.nan),
        ),
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "da",
            "max_contrast_top1_accuracy_pairwise_filtered",
            data.get("max_contrast_recovery", {}).get("top1_accuracy_pairwise_filtered", np.nan),
            n=data.get("max_contrast_recovery", {}).get("n_pairwise_filtered", np.nan),
        ),
    ]


def collect_da_rows(sweep_ids, include_references=True):
    rows = []
    if include_references:
        for model_id, spec in REFERENCE_MODELS.items():
            rows.extend(_collect_one_da(model_id, "reference", spec["label"], spec["da_path"]))

    for sweep_id in sweep_ids:
        path = (
            REPO_ROOT
            / "performance_metrics"
            / "differential_accessibility"
            / f"da_benchmark_adapter_{sweep_id}.json"
        )
        rows.extend(_collect_one_da(sweep_id, "lora_adapter", sweep_id, path))
    return rows


def _collect_one_pairwise(model_id, model_type, label, path):
    data = load_json(path)
    if data is None:
        return []
    pearson = pd.DataFrame(data["pearson"]).astype(float)
    mask = ~np.eye(len(pearson), dtype=bool)
    return [
        _row(
            model_id,
            model_type,
            label,
            "ALL",
            "pairwise_counts",
            "mean_offdiag_predicted_count_pearson",
            float(np.nanmean(pearson.values[mask])),
            float(np.nanstd(pearson.values[mask], ddof=1)),
            int(mask.sum()),
        )
    ]


def collect_pairwise_rows(sweep_ids, include_references=True):
    rows = []
    if include_references:
        for model_id, spec in REFERENCE_MODELS.items():
            rows.extend(_collect_one_pairwise(model_id, "reference", spec["label"], spec["pairwise_path"]))

    for sweep_id in sweep_ids:
        path = (
            REPO_ROOT
            / "performance_metrics"
            / "adapters"
            / sweep_id
            / f"pairwise_correlations_predicted_adapter_{sweep_id}.json"
        )
        rows.extend(_collect_one_pairwise(sweep_id, "lora_adapter", sweep_id, path))
    return rows


def plot_metric_grid(summary, benchmark, metrics, out_path, title):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42

    df = summary[summary["benchmark"] == benchmark]
    if df.empty:
        print(f"Warning: no {benchmark} rows available for {out_path}", file=sys.stderr)
        return
    pivot = (
        df[df["metric"].isin(metrics)]
        .groupby(["model_id", "label", "metric"], as_index=False)["value"]
        .mean()
        .pivot(index="metric", columns="label", values="value")
        .reindex(metrics)
    )
    if pivot.empty:
        print(f"Warning: no selected metrics available for {out_path}", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(pivot.columns)), 0.45 * len(metrics) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cross_loci(summary, out_dir):
    plot_metric_grid(
        summary,
        benchmark="cross_loci",
        metrics=[
            "profile_pearson",
            "profile_jsd",
            "log_counts_pearson",
            "counts_spearman",
        ],
        out_path=out_dir / "lora_cross_loci_metrics.pdf",
        title="LoRA adapter cross-loci benchmark",
    )


def plot_da(summary, out_dir):
    plot_metric_grid(
        summary,
        benchmark="da",
        metrics=[
            "mean_log2fc_pearson",
            "mean_log2fc_pearson_tau_filtered",
            "mean_pairwise_pearson_filtered",
            "mean_pairwise_direction_balanced_accuracy",
            "tau_auprc_all_peaks",
            "specificity_accuracy_pairwise_filtered",
            "top2_specificity_accuracy_pairwise_filtered",
            "max_contrast_top1_accuracy_pairwise_filtered",
        ],
        out_path=out_dir / "lora_da_metrics.pdf",
        title="LoRA adapter differential accessibility benchmark",
    )


def plot_pairwise(summary, out_dir):
    plot_metric_grid(
        summary,
        benchmark="pairwise_counts",
        metrics=["mean_offdiag_predicted_count_pearson"],
        out_path=out_dir / "lora_pairwise_count_metrics.pdf",
        title="LoRA adapter predicted count correlation benchmark",
    )


def write_summary_and_plots(args, sweep_ids):
    rows = []
    rows.extend(collect_cross_loci_rows(sweep_ids, args.pseudobulks, args.include_references))
    rows.extend(collect_da_rows(sweep_ids, args.include_references))
    rows.extend(collect_pairwise_rows(sweep_ids, args.include_references))

    if not rows:
        print("Warning: no LoRA benchmark outputs found to summarize.", file=sys.stderr)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "lora_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    if args.summary_only:
        return

    plot_cross_loci(summary, args.out_dir)
    plot_da(summary, args.out_dir)
    plot_pairwise(summary, args.out_dir)

    if args.plot_scatter:
        for sweep_id in sweep_ids:
            peaks_path = (
                REPO_ROOT
                / "performance_metrics"
                / "differential_accessibility"
                / f"da_benchmark_peaks_adapter_{sweep_id}.csv"
            )
            if not peaks_path.exists():
                print(
                    f"Warning: skipping scatter for {sweep_id}; missing {peaks_path}",
                    file=sys.stderr,
                )
                continue
            run_command(
                " ".join(
                    [
                        "python",
                        str(BENCHMARK_DIR / "plot_celltype_scatter.py"),
                        "--adapter-sweep-id",
                        sweep_id,
                        "--out-dir",
                        str(args.out_dir),
                    ]
                )
            )


def run_command(command):
    import subprocess

    print(command)
    result = subprocess.run(command.split())
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sweep-id",
        action="append",
        default=None,
        help="Adapter sweep ID to benchmark; repeatable. Defaults to discovered sweeps.",
    )
    parser.add_argument(
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        choices=PSEUDOBULKS,
        default=None,
        help="Pseudobulk to include in cross-loci benchmark; repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "performance_metrics" / "plots" / "lora_sweep",
    )
    parser.add_argument(
        "--plot-scatter",
        action="store_true",
        help="Also render per-sweep DA cell-type scatter plots from --save-output CSVs.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write lora_sweep_summary.csv without rendering PDF plots.",
    )
    parser.add_argument(
        "--no-references",
        dest="include_references",
        action="store_false",
        help="Exclude base and full-model finetuned references from summaries/plots.",
    )
    parser.set_defaults(include_references=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    args.pseudobulks = args.pseudobulks or PSEUDOBULKS
    sweep_ids = args.sweep_id or discover_sweep_ids()
    if not sweep_ids:
        print(
            "Error: no adapter sweep IDs provided or discovered under models/*/adapters/ "
            "or performance_metrics/adapters/.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_summary_and_plots(args, sweep_ids)


if __name__ == "__main__":
    main()
