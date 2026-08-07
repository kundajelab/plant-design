"""Shared species-namespacing helpers for the multi-species ChromBPNet pipeline.

Species are identified by Latin binomial (e.g. S.bicolor, A.thaliana, O.sativa).
Selects the active species via --species / $SPECIES (default: S.bicolor, i.e.
sorghum) and resolves config/data/output directories for it. The default
species keeps today's flat, un-namespaced paths so existing sorghum runs are
unaffected; every other species is namespaced under a same-named subdirectory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPECIES = "S.bicolor"


def add_species_arg(parser) -> None:
    parser.add_argument(
        "--species",
        default=None,
        help="Species namespace, as a Latin binomial (default: $SPECIES env var, or 'S.bicolor').",
    )


def resolve_species(cli_value: str | None = None) -> str:
    return cli_value or os.environ.get("SPECIES", DEFAULT_SPECIES)


def _ns(kind: str, species: str) -> Path:
    root = REPO_ROOT / kind
    return root if species == DEFAULT_SPECIES else root / species


def config_dir(species: str) -> Path:
    return _ns("configs", species)


def data_dir(species: str) -> Path:
    return _ns("data", species)


def models_dir(species: str) -> Path:
    return _ns("models", species)


def attributions_dir(species: str) -> Path:
    return _ns("attributions", species)


def modisco_dir(species: str) -> Path:
    return _ns("modisco", species)


def predictions_dir(species: str) -> Path:
    return _ns("predictions", species)


def performance_metrics_dir(species: str) -> Path:
    return _ns("performance_metrics", species)


def logs_dir(species: str) -> Path:
    return _ns("logs", species)


def _species_manifest(species: str) -> dict:
    manifest_path = config_dir(species) / "species.json"
    with open(manifest_path) as f:
        return json.load(f)


def genome_fasta(species: str) -> Path:
    return data_dir(species) / _species_manifest(species)["genome_fasta"]


def chrom_sizes(species: str) -> Path:
    return data_dir(species) / _species_manifest(species)["chrom_sizes"]


def fold_path(species: str, fold: int) -> Path:
    return config_dir(species) / f"fold_{fold}.json"


def read_pseudobulks(species: str, dataset: str = "base") -> list[str]:
    suffix = "" if dataset == "base" else f"_{dataset}"
    path = config_dir(species) / f"pseudobulks{suffix}.txt"
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
