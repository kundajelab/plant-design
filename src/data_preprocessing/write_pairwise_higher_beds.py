#!/usr/bin/env python3
"""Write per-cell-type DA BED files from da_peaks.csv.

For a pseudobulk P, pairwise-higher outputs include peaks where any pairwise
contrast involving P has P higher than the other cell type by at least the
threshold. Specific-union outputs add tau-specific peaks for P.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DA_DIR = REPO_ROOT / "data" / "peaks" / "differential_accessibility"
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]


def pairwise_higher_mask(da, pseudobulk, threshold):
    mask = pd.Series(False, index=da.index)
    for other in PSEUDOBULKS:
        if other == pseudobulk:
            continue

        forward = f"{pseudobulk}_vs_{other}"
        reverse = f"{other}_vs_{pseudobulk}"
        if forward in da.columns:
            mask |= da[forward] >= threshold
        elif reverse in da.columns:
            mask |= da[reverse] <= -threshold
        else:
            raise ValueError(
                f"Missing pairwise column for {pseudobulk} vs {other}: "
                f"expected {forward} or {reverse}"
            )
    return mask


def tau_specific_mask(da, pseudobulk, threshold):
    required = {"tau", "specific_to"}
    missing = required - set(da.columns)
    if missing:
        raise ValueError(f"da_peaks table missing required columns: {sorted(missing)}")
    return (da["tau"] >= threshold) & (da["specific_to"] == pseudobulk)


def write_da_set_beds(da_path, out_dir, log2fc_threshold, tau_threshold):
    da = pd.read_csv(da_path)
    required = {"chrom", "start", "end"}
    missing = required - set(da.columns)
    if missing:
        raise ValueError(f"{da_path} missing required columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for pseudobulk in PSEUDOBULKS:
        pairwise_mask = pairwise_higher_mask(da, pseudobulk, log2fc_threshold)
        tau_mask = tau_specific_mask(da, pseudobulk, tau_threshold)
        union_mask = pairwise_mask | tau_mask

        pairwise_path = out_dir / f"da_peaks_pairwise_higher_{pseudobulk}.bed"
        da.loc[pairwise_mask, ["chrom", "start", "end"]].assign(name=pseudobulk).to_csv(
            pairwise_path, sep="\t", header=False, index=False
        )
        union_path = out_dir / f"da_peaks_specific_union_{pseudobulk}.bed"
        da.loc[union_mask, ["chrom", "start", "end"]].assign(name=pseudobulk).to_csv(
            union_path, sep="\t", header=False, index=False
        )
        counts[pseudobulk] = {
            "pairwise_higher": int(pairwise_mask.sum()),
            "tau_specific": int(tau_mask.sum()),
            "specific_union": int(union_mask.sum()),
            "overlap": int((pairwise_mask & tau_mask).sum()),
        }
        print(
            f"{pairwise_path}: {counts[pseudobulk]['pairwise_higher']} peaks; "
            f"{union_path}: {counts[pseudobulk]['specific_union']} peaks "
            f"({counts[pseudobulk]['tau_specific']} tau-specific, "
            f"{counts[pseudobulk]['overlap']} overlap)"
        )
    return counts


def write_pairwise_higher_beds(da_path, out_dir, threshold):
    da = pd.read_csv(da_path)
    required = {"chrom", "start", "end"}
    missing = required - set(da.columns)
    if missing:
        raise ValueError(f"{da_path} missing required columns: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for pseudobulk in PSEUDOBULKS:
        mask = pairwise_higher_mask(da, pseudobulk, threshold)
        out_path = out_dir / f"da_peaks_pairwise_higher_{pseudobulk}.bed"
        da.loc[mask, ["chrom", "start", "end"]].assign(name=pseudobulk).to_csv(
            out_path, sep="\t", header=False, index=False
        )
        counts[pseudobulk] = int(mask.sum())
        print(f"{out_path}: {counts[pseudobulk]} peaks")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--da-peaks-csv",
        type=Path,
        default=DA_DIR / "da_peaks.csv",
        help="Path to da_peaks.csv (default: data/peaks/differential_accessibility/da_peaks.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DA_DIR,
        help="Output directory for per-cell-type BED files",
    )
    parser.add_argument(
        "--log2fc-threshold",
        type=float,
        default=1.5,
        help="Minimum directional pairwise log2FC for inclusion (default: 1.5)",
    )
    parser.add_argument(
        "--tau-threshold",
        type=float,
        default=0.75,
        help="Minimum tau for tau-specific union inclusion (default: 0.75)",
    )
    args = parser.parse_args()

    if not args.da_peaks_csv.exists():
        print(f"Error: da_peaks.csv not found: {args.da_peaks_csv}", file=sys.stderr)
        print("Run differential_accessibility.py first.", file=sys.stderr)
        sys.exit(1)

    write_da_set_beds(
        args.da_peaks_csv, args.out_dir, args.log2fc_threshold, args.tau_threshold
    )


if __name__ == "__main__":
    main()
