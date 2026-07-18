"""Dataset registry for ChromBPNet preprocessing and analysis paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DATASET_NAMES = ("base", "fine", "coarse")


@dataclass(frozen=True)
class ChromBPNetDataset:
    name: str
    pseudobulks_path: Path
    bigwig_dir: Path
    peaks_dir: Path
    nonpeaks_dir: Path
    model_root: Path
    output_suffix: str

    def pseudobulks(self) -> list[str]:
        return [
            line.strip()
            for line in self.pseudobulks_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    def validate_pseudobulk(self, pseudobulk: str) -> None:
        choices = self.pseudobulks()
        if pseudobulk not in choices:
            joined = ", ".join(choices)
            raise ValueError(
                f"Unknown pseudobulk '{pseudobulk}' for dataset '{self.name}'. "
                f"Choose one of: {joined}"
            )

    def bigwig_path(self, pseudobulk: str) -> Path:
        return self.bigwig_dir / f"{pseudobulk}.bw"

    def peaks_path(self, pseudobulk: str, *, acgt: bool = False) -> Path:
        suffix = "_peaks_ACGT.narrowPeak" if acgt else "_peaks.narrowPeak"
        return self.peaks_dir / f"{pseudobulk}{suffix}"

    def nonpeaks_path(self, pseudobulk: str, fold: int) -> Path:
        return self.nonpeaks_dir / f"{pseudobulk}_fold_{fold}_negatives.bed"

    def model_dir(self, pseudobulk: str) -> Path:
        return self.model_root / pseudobulk

    def output_subdir(self, base: str) -> str:
        return f"{base}_{self.output_suffix}" if self.output_suffix else base

    def output_path(self, root: Path, base: str) -> Path:
        return root / self.output_subdir(base)


def get_dataset(repo_root: Path, name: str) -> ChromBPNetDataset:
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {', '.join(DATASET_NAMES)}")

    configs = repo_root / "configs"
    data = repo_root / "data"
    if name == "base":
        return ChromBPNetDataset(
            name=name,
            pseudobulks_path=configs / "pseudobulks.txt",
            bigwig_dir=data / "bigwigs",
            peaks_dir=data / "peaks",
            nonpeaks_dir=data / "nonpeaks",
            model_root=repo_root / "models",
            output_suffix="",
        )
    return ChromBPNetDataset(
        name=name,
        pseudobulks_path=configs / f"pseudobulks_{name}.txt",
        bigwig_dir=data / f"bigwigs_{name}",
        peaks_dir=data / f"peaks_{name}",
        nonpeaks_dir=data / f"nonpeaks_{name}",
        model_root=repo_root / "models" / name,
        output_suffix=name,
    )


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="base",
        help="Dataset namespace to use for peaks, bigwigs, models, and outputs.",
    )
