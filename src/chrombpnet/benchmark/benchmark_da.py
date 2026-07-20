"""
Benchmark ChromBPNet model predictions against observed differential accessibility.

For each cell type, loads the 5-fold trained models and generates out-of-fold
predicted counts over the union peak set. Predicted counts are normalized and
converted to the same specificity metrics as the observed data (log2FC, pairwise
log2FC, tau, specific_to), then correlated against the observed values from
da_peaks.csv.

Requires da_peaks.csv to already exist (run differential_accessibility.py first).

Usage:
    uv run python src/chrombpnet/benchmark/benchmark_da.py
    uv run python src/chrombpnet/benchmark/benchmark_da.py --save-output --finetuned
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from daicon import load_model_with_adapter  # noqa: E402
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402

# Import DA metric functions from the preprocessing script
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent / "data_preprocessing")
)
from differential_accessibility import (
    compute_log2fc,
    compute_pairwise_log2fc,
    compute_tau,
    normalize_cpm,
)

PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
DA_PATH = REPO_ROOT / "data" / "peaks" / "differential_accessibility" / "da_peaks.csv"
IN_WINDOW = 2114
OUT_WINDOW = 1000


def corr(x, y):
    """Pearson and Spearman correlation between two 1-D arrays, NaN-safe."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    r, _ = pearsonr(x[mask], y[mask])
    rho, _ = spearmanr(x[mask], y[mask])
    return float(r), float(rho)


def safe_roc_auc(y_true, y_score):
    """AUROC with NaN handling and one-class guards."""
    mask = np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask].astype(int)
    y_score = np.asarray(y_score)[mask]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score):
    """Average precision with NaN handling and one-class guards."""
    mask = np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask].astype(int)
    y_score = np.asarray(y_score)[mask]
    if len(y_true) < 2 or y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def regression_errors(pred, obs):
    """MAE and RMSE for finite predicted/observed values."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() == 0:
        return float("nan"), float("nan")
    err = pred[mask] - obs[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def calibration_slope(pred, obs):
    """Slope from obs_logFC ~ pred_logFC through an intercept."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 2 or np.nanvar(pred[mask]) == 0:
        return float("nan")
    slope, _intercept = np.polyfit(pred[mask], obs[mask], 1)
    return float(slope)


def direction_concordance(pred, obs):
    """Sign agreement and balanced sign accuracy for non-zero observed effects."""
    mask = np.isfinite(pred) & np.isfinite(obs) & (obs != 0)
    if mask.sum() == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "positive_accuracy": float("nan"),
            "negative_accuracy": float("nan"),
            "n": 0,
            "n_positive": 0,
            "n_negative": 0,
        }

    pred_sign = np.sign(pred[mask])
    obs_sign = np.sign(obs[mask])
    agreement = pred_sign == obs_sign
    pos_mask = obs_sign > 0
    neg_mask = obs_sign < 0
    pos_acc = float(agreement[pos_mask].mean()) if pos_mask.any() else float("nan")
    neg_acc = float(agreement[neg_mask].mean()) if neg_mask.any() else float("nan")
    finite_accs = [x for x in (pos_acc, neg_acc) if np.isfinite(x)]
    balanced = float(np.mean(finite_accs)) if finite_accs else float("nan")

    return {
        "accuracy": float(agreement.mean()),
        "balanced_accuracy": balanced,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "n": int(mask.sum()),
        "n_positive": int(pos_mask.sum()),
        "n_negative": int(neg_mask.sum()),
    }


def precision_at_k(y_true, y_score, k):
    """Precision among the top-k finite scores."""
    mask = np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask].astype(int)
    y_score = np.asarray(y_score)[mask]
    if len(y_true) == 0 or k <= 0:
        return float("nan")
    k = min(int(k), len(y_true))
    order = np.argsort(y_score)[::-1][:k]
    return float(y_true[order].mean())


def ranking_metrics(y_true, y_score, ks=(100, 500, 1000)):
    """Precision@K and AP for recovering binary DA positives."""
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    metrics = {
        "average_precision": safe_average_precision(y_true, y_score),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "n_positive": n_pos,
    }
    for k in ks:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    metrics["precision_at_n_positives"] = precision_at_k(y_true, y_score, n_pos)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Pseudocount added to predicted CPM before log2FC (default: 1.0)",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save per-peak predicted and observed values to CSV",
    )
    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=0.75,
        help="Tau threshold defining cell-type-specific peaks (default: 0.75)",
    )
    parser.add_argument(
        "--log2fc-threshold",
        type=float,
        default=1.5,
        help="Max pairwise |log2FC| threshold defining pairwise DA peaks (default: 2.0)",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned models instead of base torch/ models",
        adapter_help="Use default daicon adapter bundles for the selected DA fraction",
    )
    args = parser.parse_args()

    variant = resolve_model_variant(args)
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]

    # Load observed DA values
    if not DA_PATH.exists():
        print(f"Error: da_peaks.csv not found at {DA_PATH}", file=sys.stderr)
        print("Run differential_accessibility.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading observed DA values from {DA_PATH}...")
    da = pd.read_csv(DA_PATH)
    n_peaks = len(da)
    print(f"  {n_peaks} union peaks")

    # Load chrom splits and verify models
    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = json.load(f)
    n_folds = len(chrom_splits)

    model_paths = {
        ps: [
            REPO_ROOT / "models" / ps / model_subdir / f"{ps}_f{fold}.torch"
            for fold in range(n_folds)
        ]
        for ps in PSEUDOBULKS
    }
    adapter_paths = (
        {
            ps: [
                REPO_ROOT
                / "models"
                / ps
                / "adapters"
                / adapter_id
                / f"{ps}_f{fold}.adapter.pt"
                for fold in range(n_folds)
            ]
            for ps in PSEUDOBULKS
        }
        if adapter_id is not None
        else {ps: [None] * n_folds for ps in PSEUDOBULKS}
    )
    for ps, paths in model_paths.items():
        for p in paths:
            if not p.exists():
                print(f"Error: model not found: {p}", file=sys.stderr)
                sys.exit(1)
    for ps, paths in adapter_paths.items():
        for p in paths:
            if p is not None and not p.exists():
                print(f"Error: adapter bundle not found: {p}", file=sys.stderr)
                sys.exit(1)

    # Write union peaks from da_peaks.csv to a temp narrowPeak file
    union_np = da[["chrom", "start", "end"]].copy()
    union_np["name"] = "."
    union_np["score"] = 0
    union_np["strand"] = "."
    union_np["signalValue"] = -1
    union_np["pValue"] = -1
    union_np["qValue"] = -1
    union_np["peak"] = ((da["end"] - da["start"]) // 2).astype(int)

    with tempfile.NamedTemporaryFile(suffix=".narrowPeak", mode="w", delete=False) as f:
        union_np.to_csv(f, sep="\t", header=False, index=False)
        tmp_path = f.name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build out-of-fold predicted count matrix: (n_peaks, n_pseudobulks)
    pred_counts = np.zeros((n_peaks, len(PSEUDOBULKS)), dtype=np.float32)

    for fold in range(n_folds):
        test_chroms = chrom_splits[str(fold)]
        print(f"\nFold {fold} — test chroms: {test_chroms}")

        fold_mask = da["chrom"].isin(test_chroms)
        fold_indices = da.index[fold_mask].tolist()
        if not fold_indices:
            print("  No union peaks on test chroms, skipping.")
            continue

        X, mask = extract_loci(
            loci=tmp_path,
            sequences=FASTA,
            chroms=test_chroms,
            in_window=IN_WINDOW,
            out_window=OUT_WINDOW,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
            verbose=args.verbose,
            return_mask=True,
        )
        valid_fold_indices = [idx for idx, m in zip(fold_indices, mask.tolist()) if m]

        for j, ps in enumerate(PSEUDOBULKS):
            print(f"  Predicting {ps}...")
            model, _manifest = load_model_with_adapter(
                model_paths[ps][fold],
                adapter_paths[ps][fold],
                map_location="cpu",
                adapter_root="accessibility",
            )
            pred = predict(
                model=model,
                X=X,
                device=device,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            pred_counts[valid_fold_indices, j] = torch.exp(pred[1]).squeeze().numpy()

    Path(tmp_path).unlink()

    # Compute predicted specificity metrics
    print("\nComputing predicted specificity metrics...")
    pred_cpm = normalize_cpm(pred_counts)
    pred_log2fc = compute_log2fc(pred_cpm, args.pseudocount)
    pred_pair_log2fc, pred_max_log2fc, pred_max_contrast = compute_pairwise_log2fc(
        pred_cpm, PSEUDOBULKS, args.pseudocount
    )
    pred_tau = compute_tau(pred_cpm)
    pred_specific_to = np.array([PSEUDOBULKS[i] for i in pred_cpm.argmax(axis=1)])

    # Extract observed values from da DataFrame
    obs_log2fc = da[[f"log2fc_{ps}" for ps in PSEUDOBULKS]].values
    obs_pair_log2fc = {
        col: da[col].values
        for col in da.columns
        if "_vs_" in col and col not in ("max_pairwise_contrast",)
    }
    obs_tau = da["tau"].values
    obs_specific_to = da["specific_to"].values

    # Peak subsets for filtered metrics
    tau_mask = np.isfinite(obs_tau) & (obs_tau >= args.tau_threshold)
    pairwise_mask = da["max_pairwise_log2fc"].values >= args.log2fc_threshold

    # Compute metrics
    print("Computing benchmark metrics...")

    # 1. Per-cell-type log2FC correlations (all peaks + tau-filtered peaks)
    log2fc_metrics = {}
    for i, ps in enumerate(PSEUDOBULKS):
        r_all, rho_all = corr(pred_log2fc[:, i], obs_log2fc[:, i])
        r_tau, rho_tau = corr(pred_log2fc[tau_mask, i], obs_log2fc[tau_mask, i])
        log2fc_metrics[ps] = {
            "pearson": r_all,
            "spearman": rho_all,
            "pearson_tau_filtered": r_tau,
            "spearman_tau_filtered": rho_tau,
        }

    # 2. Pairwise log2FC correlations (all peaks + pairwise-DA peaks)
    pairwise_metrics = {}
    for key, obs_vals in obs_pair_log2fc.items():
        pred_vals = pred_pair_log2fc[key]
        r_all, rho_all = corr(pred_vals, obs_vals)
        r_pw, rho_pw = corr(pred_vals[pairwise_mask], obs_vals[pairwise_mask])
        pairwise_metrics[key] = {
            "pearson": r_all,
            "spearman": rho_all,
            "pearson_pairwise_filtered": r_pw,
            "spearman_pairwise_filtered": rho_pw,
        }

    # 2b. Contrast-specific DA metrics.
    pairwise_contrast_filtered = {}
    pairwise_direction = {}
    pairwise_detection = {}
    pairwise_calibration = {}
    for key, obs_vals in obs_pair_log2fc.items():
        pred_vals = pred_pair_log2fc[key]
        finite_mask_pair = np.isfinite(obs_vals) & np.isfinite(pred_vals)
        contrast_mask = finite_mask_pair & (np.abs(obs_vals) >= args.log2fc_threshold)
        r_cf, rho_cf = corr(pred_vals[contrast_mask], obs_vals[contrast_mask])
        mae, rmse = regression_errors(pred_vals[contrast_mask], obs_vals[contrast_mask])

        positives = (
            np.abs(obs_vals[finite_mask_pair]) >= args.log2fc_threshold
        ).astype(int)
        scores = np.abs(pred_vals[finite_mask_pair])
        pairwise_contrast_filtered[key] = {
            "pearson": r_cf,
            "spearman": rho_cf,
            "n": int(contrast_mask.sum()),
        }
        pairwise_direction[key] = direction_concordance(
            pred_vals[contrast_mask], obs_vals[contrast_mask]
        )
        pairwise_detection[key] = {
            "auroc": safe_roc_auc(positives, scores),
            "auprc": safe_average_precision(positives, scores),
            "positive_rate": float(positives.mean())
            if len(positives)
            else float("nan"),
            "n_positive": int(positives.sum()),
            "n": int(len(positives)),
        }
        pairwise_calibration[key] = {
            "mae": mae,
            "rmse": rmse,
            "slope": calibration_slope(
                pred_vals[contrast_mask], obs_vals[contrast_mask]
            ),
            "n": int(contrast_mask.sum()),
        }

    pairwise_keys = list(obs_pair_log2fc.keys())
    pred_pair_abs = np.column_stack(
        [np.abs(pred_pair_log2fc[k]) for k in pairwise_keys]
    )
    pred_top1_contrast = np.array(pairwise_keys)[pred_pair_abs.argmax(axis=1)]
    pred_top2_contrast = np.array(pairwise_keys)[
        np.argsort(pred_pair_abs, axis=1)[:, -2:]
    ]
    obs_max_contrast = da["max_pairwise_contrast"].values
    max_top1 = pred_top1_contrast == obs_max_contrast
    max_top2 = np.array(
        [
            obs in set(pred_row)
            for obs, pred_row in zip(obs_max_contrast, pred_top2_contrast)
        ]
    )
    max_contrast_recovery = {
        "top1_accuracy": float(max_top1.mean()),
        "top2_accuracy": float(max_top2.mean()),
        "top1_accuracy_pairwise_filtered": (
            float(max_top1[pairwise_mask].mean())
            if pairwise_mask.any()
            else float("nan")
        ),
        "top2_accuracy_pairwise_filtered": (
            float(max_top2[pairwise_mask].mean())
            if pairwise_mask.any()
            else float("nan")
        ),
        "n": int(len(obs_max_contrast)),
        "n_pairwise_filtered": int(pairwise_mask.sum()),
    }

    ranking = {
        "celltype_tau": {},
        "pairwise_abs_log2fc": {},
    }
    for i, ps in enumerate(PSEUDOBULKS):
        positives = (tau_mask & (obs_specific_to == ps)).astype(int)
        ranking["celltype_tau"][ps] = ranking_metrics(positives, pred_log2fc[:, i])
    for key, obs_vals in obs_pair_log2fc.items():
        positives = (
            np.isfinite(obs_vals) & (np.abs(obs_vals) >= args.log2fc_threshold)
        ).astype(int)
        ranking["pairwise_abs_log2fc"][key] = ranking_metrics(
            positives, np.abs(pred_pair_log2fc[key])
        )

    # 3. Tau correlations (all peaks, NaN-safe; and pairwise-DA peaks)
    tau_r_all, tau_rho_all = corr(pred_tau, obs_tau)
    tau_r_pw, tau_rho_pw = corr(pred_tau[pairwise_mask], obs_tau[pairwise_mask])
    tau_metrics = {
        "pearson": tau_r_all,
        "spearman": tau_rho_all,
        "pearson_pairwise_filtered": tau_r_pw,
        "spearman_pairwise_filtered": tau_rho_pw,
    }

    # 4. Tau AUC / auPRC — treat tau >= threshold as positive class, predicted tau as score
    finite_mask = np.isfinite(obs_tau) & np.isfinite(pred_tau)
    pw_finite_mask = pairwise_mask & finite_mask

    def _tau_binary(mask):
        return (obs_tau[mask] >= args.tau_threshold).astype(int)

    tau_auc_metrics = {
        "auroc_all_peaks": safe_roc_auc(
            _tau_binary(finite_mask), pred_tau[finite_mask]
        ),
        "auroc_pairwise_filtered": safe_roc_auc(
            _tau_binary(pw_finite_mask), pred_tau[pw_finite_mask]
        ),
        "auprc_all_peaks": safe_average_precision(
            _tau_binary(finite_mask), pred_tau[finite_mask]
        ),
        "auprc_pairwise_filtered": safe_average_precision(
            _tau_binary(pw_finite_mask), pred_tau[pw_finite_mask]
        ),
        "positive_rate": float(_tau_binary(finite_mask).mean()),
    }

    # 5. Specificity accuracy (all peaks, tau-specific peaks, pairwise-DA peaks)
    acc_all = float((pred_specific_to == obs_specific_to).mean())
    acc_tau = (
        float((pred_specific_to[tau_mask] == obs_specific_to[tau_mask]).mean())
        if tau_mask.any()
        else float("nan")
    )
    acc_pw = (
        float(
            (pred_specific_to[pairwise_mask] == obs_specific_to[pairwise_mask]).mean()
        )
        if pairwise_mask.any()
        else float("nan")
    )
    specificity_metrics = {
        "all_peaks": acc_all,
        "tau_filtered": acc_tau,
        "pairwise_filtered": acc_pw,
        "n_tau_filtered": int(tau_mask.sum()),
        "n_pairwise_filtered": int(pairwise_mask.sum()),
        "baseline": round(1 / len(PSEUDOBULKS), 4),
    }

    # 6. Top-2 specificity accuracy: obs specific_to is within the 2 highest predicted CPM cell types
    top2_pred_indices = np.argsort(pred_cpm, axis=1)[:, -2:]
    in_top2 = np.array(
        [
            obs in [PSEUDOBULKS[idx] for idx in row]
            for obs, row in zip(obs_specific_to, top2_pred_indices)
        ]
    )
    top2_specificity_metrics = {
        "all_peaks": float(in_top2.mean()),
        "tau_filtered": float(in_top2[tau_mask].mean())
        if tau_mask.any()
        else float("nan"),
        "pairwise_filtered": float(in_top2[pairwise_mask].mean())
        if pairwise_mask.any()
        else float("nan"),
        "baseline": round(2 / len(PSEUDOBULKS), 4),
    }

    # Print summary
    print(
        f"\nPer-cell-type log2FC vs. mean of others (Pearson | Spearman)\n"
        f"  Predicted vs. observed correlation — all peaks / tau>={args.tau_threshold} peaks:"
    )
    for ps, m in log2fc_metrics.items():
        print(
            f"  {ps}: {m['pearson']:.4f} | {m['spearman']:.4f}  /  "
            f"{m['pearson_tau_filtered']:.4f} | {m['spearman_tau_filtered']:.4f}"
        )

    def _nanmean_metric(metrics, key):
        vals = [m[key] for m in metrics.values() if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(
        f"\nContrast-specific pairwise DA metrics (|obs log2FC|>={args.log2fc_threshold}):"
    )
    print(
        f"  Mean Pearson: {_nanmean_metric(pairwise_contrast_filtered, 'pearson'):.4f}"
    )
    print(
        "  Mean balanced direction accuracy: "
        f"{_nanmean_metric(pairwise_direction, 'balanced_accuracy'):.4f}"
    )
    print(
        "  Mean calibration slope: "
        f"{_nanmean_metric(pairwise_calibration, 'slope'):.4f}"
    )
    print(
        "  Max-contrast recovery top-1/top-2: "
        f"{max_contrast_recovery['top1_accuracy']:.4f} / "
        f"{max_contrast_recovery['top2_accuracy']:.4f}"
    )

    print(
        f"\nTau classification (tau>={args.tau_threshold} as positive, positive rate={tau_auc_metrics['positive_rate']:.4f})"
        f" — all / pairwise|log2FC|>={args.log2fc_threshold}:"
    )
    print(
        f"  AUROC:  {tau_auc_metrics['auroc_all_peaks']:.4f}  /  {tau_auc_metrics['auroc_pairwise_filtered']:.4f}"
    )
    print(
        f"  auPRC:  {tau_auc_metrics['auprc_all_peaks']:.4f}  /  {tau_auc_metrics['auprc_pairwise_filtered']:.4f}"
    )

    n_tau = specificity_metrics["n_tau_filtered"]
    n_pw = specificity_metrics["n_pairwise_filtered"]
    print(
        f"\nSpecificity accuracy"
        f"  (top-1 baseline {specificity_metrics['baseline']}"
        f" | top-2 baseline {top2_specificity_metrics['baseline']}):"
    )
    print(f"  {'':35s}  top-1   top-2")
    print(
        f"  {'All peaks':35s}  {specificity_metrics['all_peaks']:.4f}  {top2_specificity_metrics['all_peaks']:.4f}"
    )
    print(
        f"  {f'Tau >= {args.tau_threshold} (n={n_tau})':35s}  {specificity_metrics['tau_filtered']:.4f}  {top2_specificity_metrics['tau_filtered']:.4f}"
    )
    print(
        f"  {f'|log2FC| >= {args.log2fc_threshold} (n={n_pw})':35s}  {specificity_metrics['pairwise_filtered']:.4f}  {top2_specificity_metrics['pairwise_filtered']:.4f}"
    )

    # Save JSON
    out_dir = REPO_ROOT / "performance_metrics" / "differential_accessibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = variant["suffix"]

    results = {
        "n_peaks": n_peaks,
        "pseudocount": args.pseudocount,
        "tau_threshold": args.tau_threshold,
        "log2fc_threshold": args.log2fc_threshold,
        "log2fc_per_celltype": log2fc_metrics,
        "pairwise_log2fc": pairwise_metrics,
        "pairwise_log2fc_contrast_filtered": pairwise_contrast_filtered,
        "pairwise_direction_concordance": pairwise_direction,
        "pairwise_da_detection": pairwise_detection,
        "pairwise_effect_size_calibration": pairwise_calibration,
        "max_contrast_recovery": max_contrast_recovery,
        "ranking_metrics": ranking,
        "tau": tau_metrics,
        "tau_auc": tau_auc_metrics,
        "specificity_accuracy": specificity_metrics,
        "top2_specificity_accuracy": top2_specificity_metrics,
    }
    json_path = out_dir / f"da_benchmark{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {json_path}")

    # Optionally save per-peak CSV
    if args.save_output:
        rows = {
            "chrom": da["chrom"],
            "start": da["start"],
            "end": da["end"],
            "obs_tau": obs_tau,
            "pred_tau": pred_tau,
            "obs_specific_to": obs_specific_to,
            "pred_specific_to": pred_specific_to,
            "obs_max_pairwise_log2fc": da["max_pairwise_log2fc"].values,
            "pred_max_pairwise_log2fc": pred_max_log2fc,
            "obs_max_pairwise_contrast": da["max_pairwise_contrast"].values,
            "pred_max_pairwise_contrast": pred_max_contrast,
        }
        for i, ps in enumerate(PSEUDOBULKS):
            rows[f"pred_cpm_{ps}"] = pred_cpm[:, i]
            rows[f"obs_log2fc_{ps}"] = obs_log2fc[:, i]
            rows[f"pred_log2fc_{ps}"] = pred_log2fc[:, i]
        for key in obs_pair_log2fc:
            rows[f"obs_{key}"] = obs_pair_log2fc[key]
            rows[f"pred_{key}"] = pred_pair_log2fc[key]

        peaks_df = pd.DataFrame(rows)
        csv_path = out_dir / f"da_benchmark_peaks{suffix}.csv"
        peaks_df.to_csv(csv_path, index=False)
        print(f"Per-peak values saved to {csv_path}")


if __name__ == "__main__":
    main()
