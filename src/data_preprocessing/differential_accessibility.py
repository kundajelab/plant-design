"""
Identify differentially accessible peaks across pseudobulk cell types.

Builds a union peak set, extracts read counts from bigwigs, normalizes
(CPM or TMM), and computes per-peak specificity scores (log2FC and tau).
No replicates exist per cell type, so output is fold-change based rather
than statistical.

Output files (performance_metrics/differential_accessibility/):
  da_peaks.csv              — all peaks with CPM, log2FC, pairwise log2FC,
                              tau, specific_to, max_pairwise_log2fc, max_pairwise_contrast
  {cell_type}_ranked.csv    — per-cell-type peaks ranked by log2FC descending
  summary.json              — metadata and n_specific_peaks per cell type

Usage:
    python src/data_preprocessing/differential_accessibility.py
    python src/data_preprocessing/differential_accessibility.py --tmm
    python src/data_preprocessing/differential_accessibility.py --log2fc-threshold 1.5
"""

import argparse
import json
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tangermeme.io import extract_loci
from write_pairwise_higher_beds import write_da_set_beds

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
IN_WINDOW = 2114
OUT_WINDOW = 1000


def build_union_peaks(pseudobulks):
    """Merge all per-pseudobulk peak files into a union of non-overlapping intervals."""
    frames = []
    for ps in pseudobulks:
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
        rows.append([chrom, start, end, ".", 0, ".", -1, -1, -1, (end - start) // 2])
    return pd.DataFrame(rows)


def extract_counts(union, pseudobulks, verbose):
    """Extract summed read counts per peak per pseudobulk from bigwigs.

    Returns (counts, mask) where mask is a boolean array over the input union
    indicating which peaks were successfully extracted.
    """
    bw_paths = [str(REPO_ROOT / f"data/bigwigs/{ps}.bw") for ps in pseudobulks]
    for path in bw_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Bigwig not found: {path}")

    with tempfile.NamedTemporaryFile(suffix=".narrowPeak", mode="w", delete=False) as f:
        union.to_csv(f, sep="\t", header=False, index=False)
        tmp_path = f.name

    _, y, mask = extract_loci(
        loci=tmp_path,
        sequences=FASTA,
        signals=bw_paths,
        in_window=IN_WINDOW,
        out_window=OUT_WINDOW,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        verbose=verbose,
        return_mask=True,
    )
    Path(tmp_path).unlink()

    # y: (n_valid_peaks, n_pseudobulks, out_window) → (n_valid_peaks, n_pseudobulks)
    return y.abs().sum(dim=-1).numpy(), mask.numpy()


def normalize_cpm(counts):
    """Counts per million normalization."""
    library_sizes = counts.sum(axis=0)  # (n_pseudobulks,)
    return counts / library_sizes * 1e6


def normalize_tmm(counts, m_trim=0.3, a_trim=0.05):
    """
    TMM (trimmed mean of M-values) normalization, following edgeR's approach.

    Returns CPM values adjusted by TMM scale factors.
    """
    library_sizes = counts.sum(axis=0)  # (n_pseudobulks,)
    n_samples = counts.shape[1]

    # Choose reference: sample whose 75th-percentile count is closest to the
    # cross-sample mean of 75th-percentile counts (edgeR default heuristic).
    p75 = np.percentile(counts, 75, axis=0)
    ref_idx = int(np.argmin(np.abs(p75 - p75.mean())))

    ref_counts = counts[:, ref_idx]
    ref_lib = library_sizes[ref_idx]

    scale_factors = np.ones(n_samples)
    for i in range(n_samples):
        if i == ref_idx:
            continue

        obs_counts = counts[:, i]
        obs_lib = library_sizes[i]

        # Keep only peaks where both samples have non-zero counts
        mask = (obs_counts > 0) & (ref_counts > 0)
        if mask.sum() < 10:
            continue

        obs_r = obs_counts[mask] / obs_lib
        ref_r = ref_counts[mask] / ref_lib

        # M-values: log2 ratio; A-values: log2 average
        m = np.log2(obs_r) - np.log2(ref_r)
        a = 0.5 * (np.log2(obs_r) + np.log2(ref_r))

        # Precision weights (inverse of asymptotic variance of log ratio)
        w = (obs_lib - obs_counts[mask]) / (obs_lib * obs_counts[mask]) + (
            ref_lib - ref_counts[mask]
        ) / (ref_lib * ref_counts[mask])
        w = 1.0 / np.where(w > 0, w, np.inf)

        # Trim M and A
        m_lo, m_hi = np.quantile(m, [m_trim / 2, 1 - m_trim / 2])
        a_lo, a_hi = np.quantile(a, [a_trim / 2, 1 - a_trim / 2])
        keep = (m >= m_lo) & (m <= m_hi) & (a >= a_lo) & (a <= a_hi)

        if keep.sum() == 0:
            continue

        # Precision-weighted mean of trimmed M-values
        scale_factors[i] = 2 ** (np.average(m[keep], weights=w[keep]))

    effective_lib_sizes = library_sizes * scale_factors
    return counts / effective_lib_sizes * 1e6


def compute_log2fc(cpm, pseudocount):
    """
    Per-peak log2 fold change for each cell type vs. mean of all others.
    """
    n = cpm.shape[1]
    norm_pc = cpm + pseudocount
    # mean of others = (sum_all - self) / (n - 1)
    sum_all = norm_pc.sum(axis=1, keepdims=True)
    mean_others = (sum_all - norm_pc) / (n - 1)
    return np.log2(norm_pc / mean_others)


def compute_pairwise_log2fc(cpm, pseudobulks, pseudocount):
    """
    Per-peak log2 fold change for all C(n, 2) pairwise cell type contrasts.

    Returns:
        pair_log2fc: dict mapping "A_vs_B" -> (n_peaks,) array
        max_log2fc:  (n_peaks,) array of max |log2FC| across all pairs
        max_contrast: (n_peaks,) array of the contrast name with max |log2FC|
    """
    norm_pc = cpm + pseudocount
    pair_log2fc = {}
    for a, b in combinations(range(len(pseudobulks)), 2):
        key = f"{pseudobulks[a]}_vs_{pseudobulks[b]}"
        pair_log2fc[key] = np.log2(norm_pc[:, a] / norm_pc[:, b])

    keys = list(pair_log2fc.keys())
    matrix = np.column_stack([pair_log2fc[k] for k in keys])  # (n_peaks, n_pairs)
    abs_matrix = np.abs(matrix)
    max_idx = abs_matrix.argmax(axis=1)
    max_log2fc = abs_matrix[np.arange(len(max_idx)), max_idx]
    max_contrast = np.array(keys)[max_idx]

    return pair_log2fc, max_log2fc, max_contrast


def compute_tau(cpm):
    """
    Tau tissue-specificity score (Yanai et al. 2005), range [0, 1].
    Peaks with all-zero signal across cell types get NaN.
    """
    peak_max = cpm.max(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        x = np.where(peak_max > 0, cpm / peak_max, np.nan)
    n = cpm.shape[1]
    tau = np.nansum(1 - x, axis=1) / (n - 1)
    tau[peak_max.squeeze() == 0] = np.nan
    return tau


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tmm",
        action="store_true",
        help="Use TMM normalization instead of CPM",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1.0,
        help="Pseudocount added to CPM before computing log2FC (default: 1.0)",
    )
    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=0.75,
        help="Minimum tau score for a peak to be included in da_peaks_tau.bed (default: 0.75)",
    )
    parser.add_argument(
        "--log2fc-threshold",
        type=float,
        default=1.5,
        help="Minimum |log2FC| for a peak to be included in da_peaks_pairwise.bed (default: 1.5)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    norm_method = "tmm" if args.tmm else "cpm"

    print("Building union of peak regions...")
    union = build_union_peaks(PSEUDOBULKS)
    print(f"  {len(union)} merged peaks across all pseudobulks")

    print("Extracting read counts from bigwigs...")
    counts, mask = extract_counts(union, PSEUDOBULKS, verbose=args.verbose)
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} peaks filtered by extract_loci (scaffolds/boundaries)")
    union = union[mask].reset_index(drop=True)
    # counts: (n_valid_peaks, n_pseudobulks)

    print(f"Normalizing ({norm_method.upper()})...")
    if args.tmm:
        cpm = normalize_tmm(counts)
    else:
        cpm = normalize_cpm(counts)

    print("Computing specificity metrics...")
    log2fc = compute_log2fc(cpm, args.pseudocount)
    pair_log2fc, max_log2fc, max_contrast = compute_pairwise_log2fc(
        cpm, PSEUDOBULKS, args.pseudocount
    )
    tau = compute_tau(cpm)
    specific_to = [PSEUDOBULKS[i] for i in cpm.argmax(axis=1)]

    # Assemble output DataFrame
    coords = union[[0, 1, 2]].copy()
    coords.columns = ["chrom", "start", "end"]
    coords = coords.reset_index(drop=True)

    cpm_df = pd.DataFrame(cpm, columns=[f"cpm_{ps}" for ps in PSEUDOBULKS])
    log2fc_df = pd.DataFrame(log2fc, columns=[f"log2fc_{ps}" for ps in PSEUDOBULKS])
    pairwise_df = pd.DataFrame(pair_log2fc)
    meta_df = pd.DataFrame(
        {
            "tau": tau,
            "specific_to": specific_to,
            "max_pairwise_log2fc": max_log2fc,
            "max_pairwise_contrast": max_contrast,
        }
    )

    da = pd.concat([coords, cpm_df, log2fc_df, pairwise_df, meta_df], axis=1)

    # Save outputs
    out_dir = REPO_ROOT / "data" / "peaks" / "differential_accessibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_path = out_dir / "da_peaks.csv"
    da.to_csv(main_path, index=False)
    print(f"  Main table saved to {main_path}")

    tau_bed_path = out_dir / "da_peaks_tau.bed"
    tau_sig = da[da["tau"] >= args.tau_threshold]
    tau_sig[["chrom", "start", "end", "specific_to"]].to_csv(
        tau_bed_path, sep="\t", header=False, index=False
    )
    print(
        f"  Tau BED file saved to {tau_bed_path} ({len(tau_sig)} peaks, tau >= {args.tau_threshold})"
    )

    pairwise_bed_path = out_dir / "da_peaks_pairwise.bed"
    sig = da[da["max_pairwise_log2fc"] >= args.log2fc_threshold]
    sig[["chrom", "start", "end", "max_pairwise_contrast"]].to_csv(
        pairwise_bed_path, sep="\t", header=False, index=False
    )
    print(
        f"  Pairwise BED file saved to {pairwise_bed_path} ({len(sig)} peaks, |log2FC| >= {args.log2fc_threshold})"
    )
    write_da_set_beds(main_path, out_dir, args.log2fc_threshold, args.tau_threshold)

    for ps in PSEUDOBULKS:
        ranked = da.sort_values(f"log2fc_{ps}", ascending=False)
        ranked_path = out_dir / f"{ps}_ranked.csv"
        ranked.to_csv(ranked_path, index=False)

    n_specific = {
        ps: int(((da["specific_to"] == ps) & (da["tau"] >= args.tau_threshold)).sum())
        for ps in PSEUDOBULKS
    }

    summary = {
        "n_union_peaks": len(da),
        "normalization_method": norm_method,
        "pseudocount": args.pseudocount,
        "tau_threshold": args.tau_threshold,
        "n_specific_peaks": n_specific,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"  Summary saved to {summary_path}")

    print("\nSpecific peaks per cell type (tau > 0.75):")
    for ps, n in n_specific.items():
        print(f"  {ps}: {n}")


if __name__ == "__main__":
    main()
