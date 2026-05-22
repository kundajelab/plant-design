"""
Generates pairwise correlations of predicted ATAC counts in union of all peak
regions across all pseudobulks. Predictions come from each fold's model applied
to its held-out test chromosomes, so every peak is predicted out-of-fold.

Usage:
    python src/chrombpnet/benchmark/pairwise_correlations_predicted.py
    python src/chrombpnet/benchmark/pairwise_correlations_predicted.py --finetuned
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
from tangermeme.io import extract_loci
from tangermeme.predict import predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from daicon import load_model_with_adapter  # noqa: E402
from model_variants import add_model_variant_args, resolve_model_variant  # noqa: E402

PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
IN_WINDOW = 2114
OUT_WINDOW = 1000


def build_merged_union_peaks():
    """Load all pseudobulk peaks and merge overlapping intervals."""
    frames = []
    for ps in PSEUDOBULKS:
        path = REPO_ROOT / f"data/peaks/{ps}_peaks.narrowPeak"
        if not path.exists():
            raise FileNotFoundError(f"Peaks file not found: {path}")
        frames.append(pd.read_csv(path, sep="\t", header=None))

    all_intervals = (
        pd.concat(frames, ignore_index=True)[[0, 1, 2]]
        .sort_values([0, 1])
        .values.tolist()
    )

    merged = []
    cur_chr, cur_start, cur_end = all_intervals[0]
    for chrom, start, end in all_intervals[1:]:
        if chrom == cur_chr and start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_chr, cur_start, cur_end))
            cur_chr, cur_start, cur_end = chrom, start, end
    merged.append((cur_chr, cur_start, cur_end))

    rows = []
    for chrom, start, end in merged:
        summit = (end - start) // 2
        rows.append([chrom, start, end, ".", 0, ".", -1, -1, -1, summit])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_model_variant_args(
        parser,
        finetuned_help="Use finetuned models from models/{pseudobulk}/finetuned/ instead of torch/",
        adapter_help="Use default daicon adapter bundles for the selected DA fraction",
    )
    args = parser.parse_args()

    variant = resolve_model_variant(args)
    model_subdir = variant["model_subdir"]
    adapter_id = variant["adapter_id"]
    suffix = variant["suffix"]

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = json.load(f)
    n_folds = len(chrom_splits)

    # Verify models exist
    model_paths = {
        ps: [
            REPO_ROOT / "models" / ps / model_subdir / f"{ps}_f{fold}.torch"
            for fold in range(n_folds)
        ]
        for ps in PSEUDOBULKS
    }
    adapter_paths = {
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
    } if adapter_id is not None else {
        ps: [None] * n_folds for ps in PSEUDOBULKS
    }
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

    print("Building merged union of peak regions...")
    union = build_merged_union_peaks()
    print(f"  {len(union)} merged peaks across all pseudobulks")

    # Write union to temp narrowPeak file
    with tempfile.NamedTemporaryFile(suffix=".narrowPeak", mode="w", delete=False) as f:
        union.to_csv(f, sep="\t", header=False, index=False)
        tmp_path = f.name

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accumulate out-of-fold predicted counts: (n_peaks, n_pseudobulks)
    counts_pred = np.zeros((len(union), len(PSEUDOBULKS)), dtype=np.float32)

    for fold in range(n_folds):
        test_chroms = chrom_splits[str(fold)]
        print(f"\nFold {fold} — test chroms: {test_chroms}")

        # Identify which union peaks fall on this fold's test chroms
        fold_mask = union[0].isin(test_chroms)
        fold_indices = union.index[fold_mask].tolist()
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
            verbose=True,
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
            pred = predict(model=model, X=X, device=device, batch_size=64, verbose=True)
            # pred[1] is log counts; exp to get counts
            counts_pred[valid_fold_indices, j] = torch.exp(pred[1]).squeeze().numpy()

    Path(tmp_path).unlink()

    # Compute all pairwise Pearson and Spearman correlations
    pearson = pd.DataFrame(index=PSEUDOBULKS, columns=PSEUDOBULKS, dtype=float)
    spearman = pd.DataFrame(index=PSEUDOBULKS, columns=PSEUDOBULKS, dtype=float)

    for i, ps1 in enumerate(PSEUDOBULKS):
        for j, ps2 in enumerate(PSEUDOBULKS):
            if i == j:
                pearson.loc[ps1, ps2] = 1.0
                spearman.loc[ps1, ps2] = 1.0
            elif i < j:
                r, _ = pearsonr(counts_pred[:, i], counts_pred[:, j])
                rho, _ = spearmanr(counts_pred[:, i], counts_pred[:, j])
                pearson.loc[ps1, ps2] = r
                pearson.loc[ps2, ps1] = r
                spearman.loc[ps1, ps2] = rho
                spearman.loc[ps2, ps1] = rho

    print("\nPairwise Pearson correlations (predicted counts over union peaks):")
    print(pearson.to_string(float_format="{:.4f}".format))

    print("\nPairwise Spearman correlations (predicted counts over union peaks):")
    print(spearman.to_string(float_format="{:.4f}".format))

    # Save results
    out_dir = (
        REPO_ROOT / "performance_metrics" / "adapters" / adapter_id
        if adapter_id is not None
        else REPO_ROOT / "performance_metrics"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_union_peaks": len(union),
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }
    out_path = out_dir / f"pairwise_correlations_predicted{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    pearson.to_csv(out_dir / f"pairwise_pearson_predicted{suffix}.csv")
    spearman.to_csv(out_dir / f"pairwise_spearman_predicted{suffix}.csv")


if __name__ == "__main__":
    main()
