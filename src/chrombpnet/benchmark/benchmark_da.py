"""
Benchmark ChromBPNet model predictions against observed differential accessibility.

For each cell type, loads the 5-fold trained models and generates out-of-fold
predicted counts over the union peak set. Predicted counts are normalized and
converted to the same specificity metrics as the observed data (log2FC, pairwise
log2FC, tau, specific_to), then correlated against the observed values from
da_peaks.csv.

Requires da_peaks.csv to already exist (run differential_accessibility.py first).

Usage:
    python src/chrombpnet/benchmark/benchmark_da.py
    python src/chrombpnet/benchmark/benchmark_da.py --save-output
    python src/chrombpnet/benchmark/benchmark_da.py --finetuned
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Use finetuned models instead of base torch/ models",
    )
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="DA fraction used during finetuning; selects model/output subdirectory (default: 0.5)",
    )
    args = parser.parse_args()

    fine_tag = "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    model_subdir = fine_tag if args.finetuned else "torch"

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
    for ps, paths in model_paths.items():
        for p in paths:
            if not p.exists():
                print(f"Error: model not found: {p}", file=sys.stderr)
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
            model = torch.load(
                model_paths[ps][fold], weights_only=False, map_location="cpu"
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

    # 1. Per-cell-type log2FC correlations (all peaks + tau-specific peaks + specific_to peaks)
    log2fc_metrics = {}
    for i, ps in enumerate(PSEUDOBULKS):
        specific_mask = obs_specific_to == ps
        r_all, rho_all = corr(pred_log2fc[:, i], obs_log2fc[:, i])
        r_tau, rho_tau = corr(pred_log2fc[tau_mask, i], obs_log2fc[tau_mask, i])
        tau_specific_mask = tau_mask & specific_mask
        r_sp, rho_sp = corr(pred_log2fc[specific_mask, i], obs_log2fc[specific_mask, i])
        r_tau_sp, rho_tau_sp = corr(pred_log2fc[tau_specific_mask, i], obs_log2fc[tau_specific_mask, i])
        log2fc_metrics[ps] = {
            "pearson": r_all,
            "spearman": rho_all,
            "pearson_tau_filtered": r_tau,
            "spearman_tau_filtered": rho_tau,
            "pearson_specific_to": r_sp,
            "spearman_specific_to": rho_sp,
            "pearson_tau_specific_to": r_tau_sp,
            "spearman_tau_specific_to": rho_tau_sp,
            "n_specific_to": int(specific_mask.sum()),
            "n_tau_specific_to": int(tau_specific_mask.sum()),
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
        "auroc_all_peaks": float(roc_auc_score(_tau_binary(finite_mask), pred_tau[finite_mask]))
            if finite_mask.sum() >= 2 else float("nan"),
        "auroc_pairwise_filtered": float(roc_auc_score(_tau_binary(pw_finite_mask), pred_tau[pw_finite_mask]))
            if pw_finite_mask.sum() >= 2 else float("nan"),
        "auprc_all_peaks": float(average_precision_score(_tau_binary(finite_mask), pred_tau[finite_mask]))
            if finite_mask.sum() >= 2 else float("nan"),
        "auprc_pairwise_filtered": float(average_precision_score(_tau_binary(pw_finite_mask), pred_tau[pw_finite_mask]))
            if pw_finite_mask.sum() >= 2 else float("nan"),
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
    in_top2 = np.array([
        obs in [PSEUDOBULKS[idx] for idx in row]
        for obs, row in zip(obs_specific_to, top2_pred_indices)
    ])
    top2_specificity_metrics = {
        "all_peaks": float(in_top2.mean()),
        "tau_filtered": float(in_top2[tau_mask].mean()) if tau_mask.any() else float("nan"),
        "pairwise_filtered": float(in_top2[pairwise_mask].mean()) if pairwise_mask.any() else float("nan"),
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

    print(f"\nTau classification (tau>={args.tau_threshold} as positive, positive rate={tau_auc_metrics['positive_rate']:.4f})"
          f" — all / pairwise|log2FC|>={args.log2fc_threshold}:")
    print(f"  AUROC:  {tau_auc_metrics['auroc_all_peaks']:.4f}  /  {tau_auc_metrics['auroc_pairwise_filtered']:.4f}")
    print(f"  auPRC:  {tau_auc_metrics['auprc_all_peaks']:.4f}  /  {tau_auc_metrics['auprc_pairwise_filtered']:.4f}")

    n_tau = specificity_metrics["n_tau_filtered"]
    n_pw = specificity_metrics["n_pairwise_filtered"]
    print(
        f"\nSpecificity accuracy"
        f"  (top-1 baseline {specificity_metrics['baseline']}"
        f" | top-2 baseline {top2_specificity_metrics['baseline']}):"
    )
    print(f"  {'':35s}  top-1   top-2")
    print(f"  {'All peaks':35s}  {specificity_metrics['all_peaks']:.4f}  {top2_specificity_metrics['all_peaks']:.4f}")
    print(f"  {f'Tau >= {args.tau_threshold} (n={n_tau})':35s}  {specificity_metrics['tau_filtered']:.4f}  {top2_specificity_metrics['tau_filtered']:.4f}")
    print(f"  {f'|log2FC| >= {args.log2fc_threshold} (n={n_pw})':35s}  {specificity_metrics['pairwise_filtered']:.4f}  {top2_specificity_metrics['pairwise_filtered']:.4f}")

    # Save JSON
    out_dir = REPO_ROOT / "performance_metrics" / "differential_accessibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{fine_tag}" if args.finetuned else ""

    results = {
        "n_peaks": n_peaks,
        "pseudocount": args.pseudocount,
        "tau_threshold": args.tau_threshold,
        "log2fc_threshold": args.log2fc_threshold,
        "log2fc_per_celltype": log2fc_metrics,
        "pairwise_log2fc": pairwise_metrics,
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
