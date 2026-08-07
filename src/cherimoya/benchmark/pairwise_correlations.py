"""
Generates pairwise counts correlations in union of all peak regions
across all pseudobulks

Usage:
    uv run python src/chrombpnet/benchmark/pairwise_correlations.py
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tangermeme.io import extract_loci

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
IN_WINDOW = 2114
OUT_WINDOW = 1000


def main():
    # Build union of all peak regions (deduplicated by chr, start, end)
    print("Building union of peak regions...")
    frames = []
    for ps in PSEUDOBULKS:
        path = REPO_ROOT / f"data/peaks/{ps}_peaks.narrowPeak"
        if not path.exists():
            raise FileNotFoundError(f"Peaks file not found: {path}")
        frames.append(pd.read_csv(path, sep="\t", header=None))

    # Merge overlapping intervals: collect (chr, start, end), sort, sweep
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

    # Build narrowPeak rows with summit at merged interval center
    rows = []
    for chrom, start, end in merged:
        summit = (end - start) // 2
        rows.append([chrom, start, end, ".", 0, ".", -1, -1, -1, summit])
    union = pd.DataFrame(rows)
    print(f"  {len(union)} merged peaks across all pseudobulks")

    bw_paths = [str(REPO_ROOT / f"data/bigwigs/{ps}.bw") for ps in PSEUDOBULKS]
    for path in bw_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Bigwig not found: {path}")

    # Write union peaks to a temp narrowPeak file for extract_loci
    with tempfile.NamedTemporaryFile(suffix=".narrowPeak", mode="w", delete=False) as f:
        union.to_csv(f, sep="\t", header=False, index=False)
        tmp_path = f.name

    # Extract signals for all pseudobulks simultaneously
    print("Extracting signals over union peaks...")
    _, y = extract_loci(
        loci=tmp_path,
        sequences=FASTA,
        signals=bw_paths,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        verbose=True,
    )
    Path(tmp_path).unlink()

    # y shape: (n_peaks, n_pseudobulks, out_window) — sum to get total counts per peak
    counts = y.abs().sum(dim=-1).numpy()  # (n_peaks, n_pseudobulks)

    # Compute all pairwise Pearson and Spearman correlations
    pearson = pd.DataFrame(index=PSEUDOBULKS, columns=PSEUDOBULKS, dtype=float)
    spearman = pd.DataFrame(index=PSEUDOBULKS, columns=PSEUDOBULKS, dtype=float)

    for i, ps1 in enumerate(PSEUDOBULKS):
        for j, ps2 in enumerate(PSEUDOBULKS):
            if i == j:
                pearson.loc[ps1, ps2] = 1.0
                spearman.loc[ps1, ps2] = 1.0
            elif i < j:
                r, _ = pearsonr(counts[:, i], counts[:, j])
                rho, _ = spearmanr(counts[:, i], counts[:, j])
                pearson.loc[ps1, ps2] = r
                pearson.loc[ps2, ps1] = r
                spearman.loc[ps1, ps2] = rho
                spearman.loc[ps2, ps1] = rho

    print("\nPairwise Pearson correlations (total read counts over union peaks):")
    print(pearson.to_string(float_format="{:.4f}".format))

    print("\nPairwise Spearman correlations (total read counts over union peaks):")
    print(spearman.to_string(float_format="{:.4f}".format))

    # Save results
    out_dir = REPO_ROOT / "performance_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_union_peaks": len(union),
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }
    out_path = out_dir / "pairwise_correlations.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {out_path}")

    pearson.to_csv(out_dir / "pairwise_pearson.csv")
    spearman.to_csv(out_dir / "pairwise_spearman.csv")


if __name__ == "__main__":
    main()
