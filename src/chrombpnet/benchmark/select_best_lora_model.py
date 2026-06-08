#!/usr/bin/env python3
"""Rank LoRA adapter sweep candidates from lora_sweep_summary.csv.

The default score emphasizes differential-accessibility performance while
keeping a smaller amount of cross-loci predictive quality in the objective.
Metrics are direction-adjusted, z-scored across candidates, then combined with
the configured weights.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_SUMMARY = (
    REPO_ROOT / "performance_metrics" / "plots" / "lora_sweep" / "lora_sweep_summary.csv"
)

DEFAULT_WEIGHTS = {
    "specificity_accuracy_pairwise_filtered": 2.0,
    "mean_pairwise_pearson_filtered": 2.0,
    "mean_log2fc_pearson_tau_filtered": 1.0,
    #"tau_auprc_all_peaks": 1.0,
    "log_counts_pearson": 0.5,
    "profile_pearson": 0.5,
    "profile_jsd": -0.5,  # lower is better, so negative weight
}


def parse_weights(raw_weights):
    weights = DEFAULT_WEIGHTS.copy()
    for raw in raw_weights or []:
        if "=" not in raw:
            raise ValueError(f"Weight must be metric=value, got: {raw}")
        metric, value = raw.split("=", 1)
        weights[metric] = float(value)
    return weights


def load_summary(path):
    if not path.exists():
        print(f"Error: summary CSV not found: {path}", file=sys.stderr)
        print(
            "Run: python src/chrombpnet/benchmark/plot_lora_sweep.py --summary-only",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(path)


def candidate_metric_table(summary, include_references=False):
    if not include_references:
        summary = summary[summary["model_type"] == "lora_adapter"]
    if summary.empty:
        print("Error: no candidates found in summary CSV.", file=sys.stderr)
        sys.exit(1)

    table = (
        summary.groupby(["model_id", "model_type", "label", "metric"], as_index=False)[
            "value"
        ]
        .mean()
        .pivot(index=["model_id", "model_type", "label"], columns="metric", values="value")
    )
    return table


def zscore(values):
    values = values.astype(float)
    mean = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / sd


def score_candidates(table, weights):
    scored = table.copy()
    scored["score"] = 0.0
    used_metrics = []

    for metric, weight in weights.items():
        if metric not in scored.columns:
            continue
        metric_values = scored[metric]
        if metric_values.notna().sum() < 2:
            continue
        scored[f"z_{metric}"] = zscore(metric_values)
        scored["score"] += weight * scored[f"z_{metric}"].fillna(0.0)
        used_metrics.append(metric)

    if not used_metrics:
        print("Error: none of the weighted metrics were available for scoring.", file=sys.stderr)
        sys.exit(1)

    return scored.sort_values("score", ascending=False), used_metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--include-references",
        action="store_true",
        help="Allow base/full-finetuned reference models to compete with LoRA candidates.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=None,
        metavar="METRIC=VALUE",
        help="Override/add a scoring weight; repeatable.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = load_summary(args.summary)
    weights = parse_weights(args.weight)
    table = candidate_metric_table(summary, include_references=args.include_references)
    ranked, used_metrics = score_candidates(table, weights)

    display_cols = ["score"] + used_metrics
    result = ranked.reset_index()[["model_id", "model_type", "label"] + display_cols]
    top = result.head(args.top_k)

    print("\nBest model:")
    best = top.iloc[0]
    print(f"  {best['label']} ({best['model_id']})")
    print(f"  score={best['score']:.4f}")
    print(f"  metrics={json.dumps({m: best[m] for m in used_metrics}, indent=2)}")

    print("\nTop candidates:")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
