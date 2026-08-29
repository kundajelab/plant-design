"""Plot base, full finetuned, and daicon adapter model performance.

Reads structured JSON outputs from benchmark.py, benchmark_da.py, and
pairwise_correlations_predicted.py, then produces comparison figures saved as
PDFs to performance_metrics/plots/.

Cross-loci error bars show ±1 SD across the 5 folds (from per_fold data in
each pseudobulk JSON). DA benchmark metrics are computed over all held-out
peaks jointly, so no per-fold breakdown is available there.

Usage:
    uv run python src/chrombpnet/benchmark/plot_benchmark.py
    uv run python src/chrombpnet/benchmark/plot_benchmark.py --pseudobulks mesophyll guard
    uv run python src/chrombpnet/benchmark/plot_benchmark.py --out-dir results/plots
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42  # editable text in PDFs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    color: str
    cross_dir: Path
    da_path: Path
    pairwise_path: Path


def adapter_spec(label, color, run_id):
    return ModelSpec(
        label,
        color,
        REPO_ROOT / "performance_metrics" / "adapters" / run_id,
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / f"da_benchmark_adapter_{run_id}.json",
        REPO_ROOT
        / "performance_metrics"
        / "adapters"
        / run_id
        / f"pairwise_correlations_predicted_adapter_{run_id}.json",
    )


MODELS = [
    ModelSpec(
        "Base",
        "#4878CF",
        REPO_ROOT / "performance_metrics" / "chrombpnet",
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark.json",
        REPO_ROOT / "performance_metrics" / "pairwise_correlations_predicted.json",
    ),
    ModelSpec(
        "Finetuned 50:50",
        "#D65F5F",
        REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned",
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark_finetuned.json",
        REPO_ROOT
        / "performance_metrics"
        / "pairwise_correlations_predicted_finetuned.json",
    ),
    ModelSpec(
        "Finetuned 80:20",
        "#6ACC65",
        REPO_ROOT / "performance_metrics" / "chrombpnet_finetuned_0.8",
        REPO_ROOT
        / "performance_metrics"
        / "differential_accessibility"
        / "da_benchmark_finetuned_0.8.json",
        REPO_ROOT
        / "performance_metrics"
        / "pairwise_correlations_predicted_finetuned_0.8.json",
    ),
    adapter_spec("Adapter 50:50", "#B47CC7", "daicon_default_da0p5"),
    adapter_spec("Adapter 80:20", "#C4AD66", "daicon_default_da0p8"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_log2fc(data: dict) -> dict:
    """Return the per-celltype log2FC dict, tolerating both key names."""
    return data.get("log2fc_per_celltype") or data.get(
        "log2fc_vs_mean_others_per_celltype", {}
    )


def grouped_bars(
    ax,
    groups,
    vals_list,
    labels,
    colors,
    *,
    errs_list=None,
    ylabel="",
    title="",
    baseline=None,
):
    """Draw a grouped bar chart with one bar per model per group.

    errs_list: optional list of per-model error arrays (same shape as vals_list),
               used as ±1 SD error bars.
    """
    n = len(vals_list)
    w = min(0.22, 0.8 / max(n, 1))
    x = np.arange(len(groups))
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * w
    for i, (vals, label, color, offset) in enumerate(
        zip(vals_list, labels, colors, offsets)
    ):
        errs = errs_list[i] if errs_list is not None else None
        ax.bar(
            x + offset,
            vals,
            w,
            label=label,
            color=color,
            yerr=errs,
            capsize=3,
            error_kw={"linewidth": 1, "ecolor": "black"},
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Baseline ({baseline:.3f})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def pairwise_metric_matrix(pairwise_dict, metric_key):
    """Build a symmetric pairwise metric matrix from A_vs_B keyed metrics."""
    mat = np.full((len(PSEUDOBULKS), len(PSEUDOBULKS)), np.nan)
    for i, ps1 in enumerate(PSEUDOBULKS):
        for j, ps2 in enumerate(PSEUDOBULKS):
            if i == j:
                continue
            key = f"{ps1}_vs_{ps2}"
            rev_key = f"{ps2}_vs_{ps1}"
            if key in pairwise_dict:
                mat[i, j] = pairwise_dict[key].get(metric_key, np.nan)
            elif rev_key in pairwise_dict:
                mat[i, j] = pairwise_dict[rev_key].get(metric_key, np.nan)
    return mat


def annotate_heatmap(ax, mat, *, fmt="{:.2f}", threshold=0.5):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j,
                    i,
                    fmt.format(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] < threshold else "black",
                )


# ---------------------------------------------------------------------------
# Figure 1: Cross-loci benchmark (with per-fold error bars)
# ---------------------------------------------------------------------------


def plot_cross_loci(pseudobulks, out_dir, verbose=False):
    all_data = [
        {ps: load_json(model.cross_dir / f"{ps}.json") for ps in pseudobulks}
        for model in MODELS
    ]
    available = [all(data[ps] is not None for ps in pseudobulks) for data in all_data]
    for model, avail in zip(MODELS, available):
        if not avail:
            print(
                f"Warning: skipping {model.label} in cross-loci figure — "
                f"missing JSONs in {model.cross_dir}",
                file=sys.stderr,
            )

    active = [(m, d) for m, avail, d in zip(MODELS, available, all_data) if avail]
    if len(active) < 2:
        print(
            "Warning: fewer than 2 model sets available; skipping cross-loci figure.",
            file=sys.stderr,
        )
        return

    labels = [m.label for m, _ in active]
    colors = [m.color for m, _ in active]
    datasets = [d for _, d in active]

    metrics = [
        ("profile_pearson", "Profile Pearson"),
        ("profile_jsd", "Profile JSD (lower=better)"),
        ("log_counts_pearson", "Log-counts Pearson"),
        ("counts_spearman", "Counts Spearman"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, metrics):
        vals_list, errs_list = [], []
        for data in datasets:
            means, stds = [], []
            for ps in pseudobulks:
                fold_vals = [
                    data[ps]["per_fold"][str(f)][key]
                    for f in sorted(int(k) for k in data[ps]["per_fold"])
                ]
                means.append(np.mean(fold_vals))
                stds.append(np.std(fold_vals, ddof=1))
            vals_list.append(means)
            errs_list.append(stds)
        grouped_bars(
            ax, pseudobulks, vals_list, labels, colors, errs_list=errs_list, title=label
        )

    # Shared legend
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, lbls, loc="upper right", bbox_to_anchor=(1.12, 1), frameon=False
    )
    fig.suptitle(
        "Cross-loci benchmark (error bars = ±1 SD across folds)", fontsize=13, y=1.01
    )
    fig.tight_layout()

    out_path = out_dir / "cross_loci.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Mean-subtracted log2FC Pearson (three peak filters)
# ---------------------------------------------------------------------------


def plot_da_log2fc(out_dir, verbose=False):
    """Two-panel figure: per-cell-type log2FC Pearson under all / tau-filtered peaks."""
    paths = [model.da_path for model in MODELS]
    all_data = [load_json(p) for p in paths]
    available = [d is not None for d in all_data]
    for model, avail, p in zip(MODELS, available, paths):
        label = model.label
        if not avail:
            print(f"Warning: {label} DA benchmark JSON not found: {p}", file=sys.stderr)
    if not available[0]:
        print("Error: base da_benchmark.json is required.", file=sys.stderr)
        sys.exit(1)

    active = [(m, d) for m, avail, d in zip(MODELS, available, all_data) if avail]
    labels = [m.label for m, _ in active]
    colors = [m.color for m, _ in active]
    datasets = [d for _, d in active]

    tau_threshold = datasets[0].get("tau_threshold", 0.75)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    grouped_bars(
        axes[0],
        PSEUDOBULKS,
        [[get_log2fc(d)[ps]["pearson"] for ps in PSEUDOBULKS] for d in datasets],
        labels,
        colors,
        title="Per-cell-type log2FC Pearson\n(all peaks)",
        ylabel="Pearson r",
    )
    grouped_bars(
        axes[1],
        PSEUDOBULKS,
        [
            [get_log2fc(d)[ps]["pearson_tau_filtered"] for ps in PSEUDOBULKS]
            for d in datasets
        ],
        labels,
        colors,
        title=f"Per-cell-type log2FC Pearson\n(tau≥{tau_threshold} peaks)",
        ylabel="Pearson r",
    )

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, lbls, loc="upper right", bbox_to_anchor=(1.12, 1), frameon=False
    )
    fig.suptitle(
        "Mean-subtracted log2FC Pearson (predicted vs observed)", fontsize=11, y=1.03
    )
    fig.tight_layout()

    out_path = out_dir / "da_log2fc.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: DA benchmark (pairwise Pearson + specificity accuracy)
# ---------------------------------------------------------------------------


def plot_da_benchmark(out_dir, verbose=False):
    paths = [model.da_path for model in MODELS]
    all_data = [load_json(p) for p in paths]

    available = [d is not None for d in all_data]
    for model, avail, p in zip(MODELS, available, paths):
        label = model.label
        if not avail:
            print(f"Warning: {label} DA benchmark JSON not found: {p}", file=sys.stderr)

    if not available[0]:
        print("Error: base da_benchmark.json is required.", file=sys.stderr)
        sys.exit(1)
    if not any(available[1:]):
        print(
            "Error: at least one non-base da_benchmark JSON is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    active = [(m, d) for m, avail, d in zip(MODELS, available, all_data) if avail]
    labels = [m.label for m, _ in active]
    datasets = [d for _, d in active]
    n_models = len(active)

    ref = datasets[0]
    log2fc_threshold = ref.get("log2fc_threshold", 1.5)
    tau_threshold = ref.get("tau_threshold", 0.75)
    n_pairwise = ref["specificity_accuracy"]["n_pairwise_filtered"]
    baseline = ref["specificity_accuracy"]["baseline"]
    top2_baseline = ref.get("top2_specificity_accuracy", {}).get(
        "baseline", round(2 / len(PSEUDOBULKS), 4)
    )

    short_labels = [ps.replace("_", "\n") for ps in PSEUDOBULKS]
    ncols = max(n_models, 2)
    fig, axes = plt.subplots(
        2, ncols, figsize=(5.5 * ncols, 11), constrained_layout=True
    )

    def pairwise_to_matrix(pairwise_dict, metric_key):
        mat = np.full((len(PSEUDOBULKS), len(PSEUDOBULKS)), np.nan)
        for i, ps1 in enumerate(PSEUDOBULKS):
            for j, ps2 in enumerate(PSEUDOBULKS):
                if i == j:
                    continue
                key = (
                    f"{ps1}_vs_{ps2}"
                    if f"{ps1}_vs_{ps2}" in pairwise_dict
                    else f"{ps2}_vs_{ps1}"
                )
                if key in pairwise_dict:
                    mat[i, j] = pairwise_dict[key][metric_key]
        return mat

    # Row 0: pairwise log2FC Pearson heatmaps (one per model)
    im = None
    for col, (model, data) in enumerate(active):
        _label = model.label
        color = model.color
        ax = axes[0, col]
        mat = pairwise_to_matrix(data["pairwise_log2fc"], "pearson_pairwise_filtered")
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("lightgray")
        im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(PSEUDOBULKS)))
        ax.set_yticks(range(len(PSEUDOBULKS)))
        ax.set_xticklabels(short_labels, fontsize=7)
        ax.set_yticklabels(short_labels, fontsize=7)
        ax.set_title(_label)
        for i in range(len(PSEUDOBULKS)):
            for j in range(len(PSEUDOBULKS)):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if mat[i, j] < 0.5 else "black",
                    )
    for col in range(n_models, ncols):
        axes[0, col].set_visible(False)
    fig.colorbar(im, ax=axes[0, :n_models].tolist(), shrink=0.7, label="Pearson r")
    axes[0, 0].set_ylabel(
        f"Pairwise log2FC Pearson\n(|log2FC|≥{log2fc_threshold}, n={n_pairwise:,})",
        fontsize=9,
    )

    # Row 1: specificity accuracy bar charts
    acc_groups = ["All peaks", f"Tau≥{tau_threshold}", f"|log2FC|≥{log2fc_threshold}"]
    acc_colors = [m.color for m, _ in active]
    grouped_bars(
        axes[1, 0],
        acc_groups,
        [
            [
                d["specificity_accuracy"]["all_peaks"],
                d["specificity_accuracy"]["tau_filtered"],
                d["specificity_accuracy"]["pairwise_filtered"],
            ]
            for d in datasets
        ],
        labels,
        acc_colors,
        title="Top-1 specificity accuracy\n(predicted vs observed cell type)",
        ylabel="Accuracy",
        baseline=baseline,
    )
    grouped_bars(
        axes[1, 1],
        acc_groups,
        [
            [
                d.get("top2_specificity_accuracy", {}).get("all_peaks", float("nan")),
                d.get("top2_specificity_accuracy", {}).get(
                    "tau_filtered", float("nan")
                ),
                d.get("top2_specificity_accuracy", {}).get(
                    "pairwise_filtered", float("nan")
                ),
            ]
            for d in datasets
        ],
        labels,
        acc_colors,
        title="Top-2 specificity accuracy\n(obs cell type in top-2 predicted)",
        ylabel="Accuracy",
        baseline=top2_baseline,
    )
    for col in range(2, ncols):
        axes[1, col].set_visible(False)

    # Shared legend
    all_handles, all_labels_leg = [], []
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            if lbl not in all_labels_leg:
                all_handles.append(h)
                all_labels_leg.append(lbl)
    fig.legend(all_handles, all_labels_leg, loc="outside upper right", frameon=False)
    fig.suptitle(
        "Differential accessibility benchmark\n(no per-fold error bars — metrics computed jointly over all held-out peaks)",
        fontsize=12,
    )

    out_path = out_dir / "da_benchmark.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Pairwise log2FC Pearson heatmaps (standalone)
# ---------------------------------------------------------------------------


def plot_da_pairwise_log2fc(out_dir, verbose=False):
    """Standalone figure: pairwise log2FC Pearson heatmaps, all peaks vs filtered."""
    paths = [model.da_path for model in MODELS]
    all_data = [load_json(p) for p in paths]
    available = [d is not None for d in all_data]
    for model, avail, p in zip(MODELS, available, paths):
        label = model.label
        if not avail:
            print(f"Warning: {label} DA benchmark JSON not found: {p}", file=sys.stderr)
    if not available[0]:
        print("Error: base da_benchmark.json is required.", file=sys.stderr)
        sys.exit(1)

    active = [(m, d) for m, avail, d in zip(MODELS, available, all_data) if avail]
    n_models = len(active)

    ref = active[0][1]
    log2fc_threshold = ref.get("log2fc_threshold", 1.5)
    n_all = ref["pairwise_log2fc"][f"{PSEUDOBULKS[0]}_vs_{PSEUDOBULKS[1]}"].get(
        "n", None
    )
    n_filtered = ref.get("specificity_accuracy", {}).get("n_pairwise_filtered", None)

    def pairwise_to_matrix(pairwise_dict, metric_key):
        return pairwise_metric_matrix(pairwise_dict, metric_key)

    short_labels = [ps.replace("_", "\n") for ps in PSEUDOBULKS]
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("lightgray")

    n_rows, n_cols = 2, n_models
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 9), squeeze=False, constrained_layout=True
    )

    row_titles = [
        "All peaks — pearson",
        f"|log2FC|≥{log2fc_threshold} — pearson (pairwise filtered)",
    ]
    metric_keys = ["pearson", "pearson_pairwise_filtered"]

    im = None
    for col, (model, data) in enumerate(active):
        model_label = model.label
        _color = model.color
        for row, (metric_key, row_title) in enumerate(zip(metric_keys, row_titles)):
            ax = axes[row, col]
            mat = pairwise_to_matrix(data["pairwise_log2fc"], metric_key)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(PSEUDOBULKS)))
            ax.set_yticks(range(len(PSEUDOBULKS)))
            ax.set_xticklabels(short_labels, fontsize=7)
            ax.set_yticklabels(short_labels, fontsize=7)
            if row == 0:
                ax.set_title(model_label, fontsize=10)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8)
            for i in range(len(PSEUDOBULKS)):
                for j in range(len(PSEUDOBULKS)):
                    if not np.isnan(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if mat[i, j] < 0.5 else "black",
                        )

    fig.colorbar(
        im, ax=axes[:, :n_models].ravel().tolist(), shrink=0.6, label="Pearson r"
    )
    fig.suptitle("Pairwise log2FC Pearson (predicted vs observed)", fontsize=12)

    out_path = out_dir / "da_pairwise_log2fc.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Contrast-specific DA metric heatmaps
# ---------------------------------------------------------------------------


def plot_da_contrast_metrics(out_dir, verbose=False):
    """Compact heatmaps for the new contrast-specific DA metrics."""
    paths = [model.da_path for model in MODELS]
    all_data = [load_json(p) for p in paths]
    available = [d is not None for d in all_data]
    for model, avail, p in zip(MODELS, available, paths):
        label = model.label
        if not avail:
            print(f"Warning: {label} DA benchmark JSON not found: {p}", file=sys.stderr)

    required = [
        "pairwise_log2fc_contrast_filtered",
        "pairwise_direction_concordance",
        "pairwise_effect_size_calibration",
    ]
    active = [
        (m, d)
        for m, avail, d in zip(MODELS, available, all_data)
        if avail and all(key in d for key in required)
    ]
    if not active:
        print(
            "Warning: no DA benchmark JSON contains contrast-specific metrics; "
            "re-run benchmark_da.py first.",
            file=sys.stderr,
        )
        return

    n_models = len(active)
    short_labels = [ps.replace("_", "\n") for ps in PSEUDOBULKS]
    rows = [
        (
            "pairwise_log2fc_contrast_filtered",
            "pearson",
            "Contrast-specific log2FC Pearson",
            0,
            1,
            matplotlib.colormaps["viridis"].copy(),
            0.5,
        ),
        (
            "pairwise_direction_concordance",
            "balanced_accuracy",
            "Balanced direction accuracy",
            0,
            1,
            matplotlib.colormaps["viridis"].copy(),
            0.5,
        ),
        (
            "pairwise_effect_size_calibration",
            "slope",
            "Calibration slope",
            0,
            2,
            matplotlib.colormaps["coolwarm"].copy(),
            1.0,
        ),
    ]
    for *_unused, cmap, _threshold in rows:
        cmap.set_bad("lightgray")

    fig, axes = plt.subplots(
        len(rows),
        n_models,
        figsize=(5 * n_models, 4.2 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )

    row_ims = [None] * len(rows)
    for col, (model, data) in enumerate(active):
        model_label = model.label
        _color = model.color
        for row_idx, (
            section,
            metric_key,
            row_title,
            vmin,
            vmax,
            cmap,
            threshold,
        ) in enumerate(rows):
            ax = axes[row_idx, col]
            mat = pairwise_metric_matrix(data[section], metric_key)
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            row_ims[row_idx] = im
            ax.set_xticks(range(len(PSEUDOBULKS)))
            ax.set_yticks(range(len(PSEUDOBULKS)))
            ax.set_xticklabels(short_labels, fontsize=7)
            ax.set_yticklabels(short_labels, fontsize=7)
            if row_idx == 0:
                ax.set_title(model_label, fontsize=10)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8)
            annotate_heatmap(ax, mat, threshold=threshold)
    for row_idx, im in enumerate(row_ims):
        fig.colorbar(im, ax=axes[row_idx, :].tolist(), shrink=0.65)

    log2fc_threshold = active[0][1].get("log2fc_threshold", 1.5)
    fig.suptitle(
        f"Contrast-specific DA metrics (|obs log2FC|≥{log2fc_threshold})",
        fontsize=12,
    )

    out_path = out_dir / "da_contrast_metrics.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Pairwise count correlation heatmaps
# ---------------------------------------------------------------------------


def plot_pairwise_heatmaps(out_dir, verbose=False):
    sources = [
        (REPO_ROOT / "performance_metrics" / "pairwise_correlations.json", "Observed"),
        *[(model.pairwise_path, model.label) for model in MODELS],
    ]
    panels = [(load_json(p), title) for p, title in sources]
    panels = [(d, title) for d, title in panels if d is not None]

    if len(panels) < 2:
        print(
            "Warning: fewer than 2 pairwise JSONs available; skipping heatmap figure.",
            file=sys.stderr,
        )
        return

    def to_matrix(corr_dict):
        return np.array(
            [[corr_dict[ps1][ps2] for ps2 in PSEUDOBULKS] for ps1 in PSEUDOBULKS]
        )

    short_labels = [ps.replace("_", "\n") for ps in PSEUDOBULKS]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, (data, title) in zip(axes, panels):
        mat = to_matrix(data["pearson"])
        im = ax.imshow(
            mat, vmin=0, vmax=1, cmap=matplotlib.colormaps["viridis"], aspect="auto"
        )
        ax.set_xticks(range(len(PSEUDOBULKS)))
        ax.set_yticks(range(len(PSEUDOBULKS)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title(title)
        for i in range(len(PSEUDOBULKS)):
            for j in range(len(PSEUDOBULKS)):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if mat[i, j] < 0.5 else "black",
                )

    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    fig.suptitle("Pairwise count correlations (Pearson)", fontsize=13)

    out_path = out_dir / "pairwise_corr.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pseudobulks",
        nargs="+",
        default=PSEUDOBULKS,
        choices=PSEUDOBULKS,
        metavar="PSEUDOBULK",
        help="Pseudobulks to include in cross-loci figure (default: all 6)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "performance_metrics" / "plots",
        help="Output directory for PDF figures (default: performance_metrics/plots)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_cross_loci(args.pseudobulks, args.out_dir, verbose=args.verbose)
    plot_da_log2fc(args.out_dir, verbose=args.verbose)
    plot_da_benchmark(args.out_dir, verbose=args.verbose)
    plot_da_pairwise_log2fc(args.out_dir, verbose=args.verbose)
    plot_da_contrast_metrics(args.out_dir, verbose=args.verbose)
    plot_pairwise_heatmaps(args.out_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
