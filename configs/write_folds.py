import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import species_config

parser = argparse.ArgumentParser()
species_config.add_species_arg(parser)
args = parser.parse_args()
species = species_config.resolve_species(args.species)

config_dir = species_config.config_dir(species)

with open(config_dir / "chrom_splits.json") as f:
    folds = json.load(f)

for i in range(5):
    p = {}
    p["test"] = folds[str(i)]
    p["valid"] = folds[str((i + 1) % 5)]
    train_folds = [str(j) for j in range(5) if j != i and j != (i + 1) % 5]
    p["train"] = []
    for f in train_folds:
        p["train"].extend(folds[f])
    with open(species_config.fold_path(species, i), "w") as f:
        json.dump(p, f, indent=4)
