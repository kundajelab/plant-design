"""Utilities for ChromBPNet DA peak attribution inputs."""

from pathlib import Path

PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
DA_PEAK_CHOICES = ["tau", "pairwise", "pairwise_higher", "specific_union"]


def da_suffix(da_peaks):
    """Return the filename suffix for a DA peak set."""
    return f"_da_{da_peaks}" if da_peaks is not None else ""


def resolve_da_bed(repo_root, da_peaks, pseudobulk):
    """Resolve a DA peak-set name to the BED path used for a pseudobulk."""
    da_dir = Path(repo_root) / "data" / "peaks" / "differential_accessibility"
    if da_peaks == "tau":
        return da_dir / "da_peaks_tau.bed"
    if da_peaks == "pairwise":
        return da_dir / "da_peaks_pairwise.bed"
    if da_peaks == "pairwise_higher":
        return da_dir / f"da_peaks_pairwise_higher_{pseudobulk}.bed"
    if da_peaks == "specific_union":
        return da_dir / f"da_peaks_specific_union_{pseudobulk}.bed"
    raise ValueError(f"unknown DA peak set: {da_peaks}")


def resolve_loci_path(repo_root, pseudobulk, da_peaks=None):
    """Resolve all-peaks or DA-peaks loci path for a pseudobulk."""
    if da_peaks is None:
        return Path(repo_root) / "data" / "peaks" / f"{pseudobulk}_peaks_ACGT.narrowPeak"
    return resolve_da_bed(repo_root, da_peaks, pseudobulk)


def resolve_ohe_path(repo_root, pseudobulk, da_peaks=None):
    """Resolve the OHE npz path matching a pseudobulk and optional DA peak set."""
    return (
        Path(repo_root)
        / "attributions"
        / "chrombpnet"
        / f"{pseudobulk}_ohe{da_suffix(da_peaks)}.npz"
    )
