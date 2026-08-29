#!/usr/bin/env python3
"""Import a species' chromosome folds from chrom_splits.yaml into configs/<species>/chrom_splits.json.

chrom_splits.yaml holds fold assignments for many species keyed by Latin
binomial (e.g. A.thaliana, O.sativa) — the same identifiers passed to
--species everywhere else in the pipeline. This writes the JSON shape that
species_config.py expects at configs/<species>/chrom_splits.json.

Usage:
    python configs/import_chrom_splits.py --species A.thaliana
    python configs/import_chrom_splits.py --species O.sativa
    python configs/import_chrom_splits.py --species rice --key O.sativa  # differing key
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import species_config


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--species", required=True, help="Species namespace to write to (e.g. A.thaliana)"
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Key in chrom_splits.yaml, if different from --species (default: --species)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "chrom_splits.yaml",
        help="Source YAML file (default: configs/chrom_splits.yaml)",
    )
    args = parser.parse_args()
    key = args.key or args.species

    with open(args.source) as f:
        all_splits = yaml.safe_load(f)
    if key not in all_splits:
        choices = ", ".join(sorted(all_splits))
        raise SystemExit(f"'{key}' not found in {args.source}. Available: {choices}")

    folds = {str(k): v for k, v in sorted(all_splits[key].items())}

    out_dir = species_config.config_dir(args.species)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chrom_splits.json"
    with open(out_path, "w") as f:
        json.dump(folds, f, indent=4)
    print(f"Wrote {len(folds)} fold(s) for '{key}' to {out_path}")


if __name__ == "__main__":
    main()
