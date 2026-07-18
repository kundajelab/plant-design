"""Utilities for ChromBPNet DA peak attribution inputs."""

from pathlib import Path

DATASETS = ("base", "fine", "coarse")
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


def model_dir(repo_root, dataset, pseudobulk):
    root = Path(repo_root) / "models"
    if dataset != "base":
        root = root / dataset
    return root / pseudobulk


def read_pseudobulks(repo_root, dataset):
    suffix = "" if dataset == "base" else f"_{dataset}"
    path = Path(repo_root) / "configs" / f"pseudobulks{suffix}.txt"
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def validate_pseudobulk(repo_root, dataset, pseudobulk):
    choices = read_pseudobulks(repo_root, dataset)
    if pseudobulk not in choices:
        raise ValueError(
            f"unknown pseudobulk '{pseudobulk}' for dataset '{dataset}'. "
            f"Choose one of: {', '.join(choices)}"
        )


def resolve_loci_path(repo_root, pseudobulk, da_peaks=None, dataset="base"):
    """Resolve all-peaks or DA-peaks loci path for a pseudobulk."""
    if da_peaks is None:
        suffix = "" if dataset == "base" else f"_{dataset}"
        return (
            Path(repo_root)
            / "data"
            / f"peaks{suffix}"
            / f"{pseudobulk}_peaks_ACGT.narrowPeak"
        )
    return resolve_da_bed(repo_root, da_peaks, pseudobulk)


def resolve_ohe_path(repo_root, pseudobulk, da_peaks=None, dataset="base"):
    """Resolve the OHE npz path matching a pseudobulk and optional DA peak set."""
    subdir = "chrombpnet" if dataset == "base" else f"chrombpnet_{dataset}"
    return (
        Path(repo_root)
        / "attributions"
        / subdir
        / f"{pseudobulk}_ohe{da_suffix(da_peaks)}.npz"
    )
