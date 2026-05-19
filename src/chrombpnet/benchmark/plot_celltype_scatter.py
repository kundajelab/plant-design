"""Plot cross-cell-type scatter grid of observed and predicted ATAC counts.

Produces a 6×6 grid of scatter plots comparing log₁₀(CPM + 1) across cell types:
  - Lower triangle: observed CPM in cell type X vs observed CPM in cell type Y
  - Diagonal:       predicted CPM vs observed CPM (same cell type)
  - Upper triangle: predicted CPM in cell type X vs predicted CPM in cell type Y

Each cell is annotated with the Pearson r on log-transformed values.

Requires:
    data/peaks/differential_accessibility/da_peaks.csv
    performance_metrics/differential_accessibility/da_benchmark_peaks{suffix}.csv
    (run differential_accessibility.py and benchmark_da.py --save-output first)

Usage:
    python src/chrombpnet/benchmark/plot_celltype_scatter.py
    python src/chrombpnet/benchmark/plot_celltype_scatter.py --finetuned
    python src/chrombpnet/benchmark/plot_celltype_scatter.py --finetuned --da-fraction 0.8
    python src/chrombpnet/benchmark/plot_celltype_scatter.py --out-dir results/plots
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

matplotlib.rcParams["pdf.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
SHORT_LABELS = {
    "bundle_sheath": "BS",
    "epidermis": "Epi",
    "guard": "GC",
    "mesophyll": "Mes",
    "phloem": "Phl",
    "xylem": "Xyl",
}

COLOR_OBS = "#4878CF"  # lower triangle — observed vs observed
COLOR_DIAG = "#1a1a2e"  # diagonal       — predicted vs observed (same cell type)
COLOR_PRED = "#9BB7D4"  # upper triangle — predicted vs predicted

PSEUDOCOUNT = 1.0  # added before log10 to handle zero-CPM peaks


def _scatter(ax, x, y, color):
    """Log-scale scatter with Pearson r annotation."""
    xp = np.log10(x + PSEUDOCOUNT)
    yp = np.log10(y + PSEUDOCOUNT)
    finite = np.isfinite(xp) & np.isfinite(yp)
    r, _ = pearsonr(xp[finite], yp[finite])
    ax.scatter(
        x[finite] + PSEUDOCOUNT,
        y[finite] + PSEUDOCOUNT,
        s=0.5,
        alpha=0.12,
        color=color,
        rasterized=True,
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(
        0.06,
        0.94,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        fontsize=5.5,
        va="top",
        ha="left",
    )
    ax.tick_params(labelsize=4.5, length=2, width=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use predictions from finetuned models",
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects input file suffix (default: 0.5)",
    )
    parser.add_argument(
        "--adapter-sweep-id",
        type=str,
        default=None,
        help="Use DA benchmark peak output for this adapter sweep ID",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "performance_metrics" / "plots",
        help="Output directory for PDF figure (default: performance_metrics/plots)",
    )
    args = parser.parse_args()

    if args.finetuned and args.adapter_sweep_id is not None:
        print(
            "Error: use either --finetuned or --adapter-sweep-id, not both.",
            file=sys.stderr,
        )
        sys.exit(1)

    fine_tag = (
        "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    )
    suffix = (
        f"_adapter_{args.adapter_sweep_id}"
        if args.adapter_sweep_id is not None
        else (f"_{fine_tag}" if args.finetuned else "")
    )

    da_path = (
        REPO_ROOT / "data" / "peaks" / "differential_accessibility" / "da_peaks.csv"
    )
    peaks_path = (
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / f"da_benchmark_peaks{suffix}.csv"
    )

    for path, label in [
        (da_path, "da_peaks.csv"),
        (peaks_path, f"da_benchmark_peaks{suffix}.csv"),
    ]:
        if not path.exists():
            print(f"Error: {label} not found: {path}", file=sys.stderr)
            if "benchmark_peaks" in str(path):
                print(
                    "Run: python src/chrombpnet/benchmark/benchmark_da.py --save-output"
                    + (" --finetuned" if args.finetuned else "")
                    + (
                        f" --adapter-sweep-id {args.adapter_sweep_id}"
                        if args.adapter_sweep_id is not None
                        else ""
                    )
                    + (
                        f" --da-fraction {args.da_fraction}"
                        if args.finetuned and args.da_fraction != 0.5
                        else ""
                    ),
                    file=sys.stderr,
                )
            sys.exit(1)

    obs_cols = [f"cpm_{ps}" for ps in PSEUDOBULKS]
    pred_cols = [f"pred_cpm_{ps}" for ps in PSEUDOBULKS]

    da = pd.read_csv(da_path, usecols=["chrom", "start", "end"] + obs_cols)
    peaks = pd.read_csv(peaks_path)

    missing = [c for c in pred_cols if c not in peaks.columns]
    if missing:
        print(
            f"Error: {peaks_path.name} is missing pred_cpm_* columns.\n"
            "Re-run benchmark_da.py with --save-output to regenerate it:\n"
            "  python src/chrombpnet/benchmark/benchmark_da.py --save-output"
            + (" --finetuned" if args.finetuned else "")
            + (
                f" --adapter-sweep-id {args.adapter_sweep_id}"
                if args.adapter_sweep_id is not None
                else ""
            )
            + (
                f" --da-fraction {args.da_fraction}"
                if args.finetuned and args.da_fraction != 0.5
                else ""
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    peaks = peaks[["chrom", "start", "end"] + pred_cols]

    merged = da.merge(peaks, on=["chrom", "start", "end"], how="inner")
    if merged.empty:
        print(
            "Error: no peaks matched between da_peaks.csv and da_benchmark_peaks.csv",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Plotting {len(merged)} peaks")

    n = len(PSEUDOBULKS)
    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), squeeze=False)

    for i, ps_row in enumerate(PSEUDOBULKS):
        for j, ps_col in enumerate(PSEUDOBULKS):
            ax = axes[i, j]

            obs_x = merged[f"cpm_{ps_col}"].values
            obs_y = merged[f"cpm_{ps_row}"].values
            pred_x = merged[f"pred_cpm_{ps_col}"].values
            pred_y = merged[f"pred_cpm_{ps_row}"].values

            if i > j:
                # Lower triangle: observed vs observed
                _scatter(ax, obs_x, obs_y, COLOR_OBS)
            elif i < j:
                # Upper triangle: predicted vs predicted
                _scatter(ax, pred_x, pred_y, COLOR_PRED)
            else:
                # Diagonal: predicted vs observed (same cell type)
                _scatter(ax, obs_x, pred_y, COLOR_DIAG)

            # Outer-edge labels only
            if i == n - 1:
                ax.set_xlabel(SHORT_LABELS[ps_col], fontsize=7, labelpad=2)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(SHORT_LABELS[ps_row], fontsize=7, labelpad=2)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    title = (
        "Cross-cell-type ATAC counts  (log₁₀ CPM + 1)\n"
        "Lower: obs vs obs   |   Diagonal: pred vs obs   |   Upper: pred vs pred"
    )
    if args.adapter_sweep_id is not None:
        title += f"   [adapter: {args.adapter_sweep_id}]"
    elif args.finetuned:
        title += f"   [{fine_tag}]"
    fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"celltype_scatter{suffix}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
