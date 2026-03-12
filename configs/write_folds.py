import json

with open("chrom_splits.json") as f:
    folds = json.load(f)

for i in range(5):
    p = {}
    p["test"] = folds[str(i)]
    p["valid"] = folds[str((i + 1) % 5)]
    train_folds = [str(j) for j in range(5) if j != i and j != (i + 1) % 5]
    p["train"] = []
    for f in train_folds:
        p["train"].extend(folds[f])
    with open(f"fold_{i}.json", "w") as f:
        json.dump(p, f, indent=4)
