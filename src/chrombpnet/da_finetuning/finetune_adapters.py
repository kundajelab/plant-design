#!/usr/bin/env python3
"""Finetune low-rank adapters for ChromBPNet differential accessibility.

The base ChromBPNet model is loaded from models/{pseudobulk}/torch/ and kept
frozen. Low-rank adapters are injected into the accessibility BPNet and saved
as separate adapter bundles under models/{pseudobulk}/adapters/{sweep_id}/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from bpnetlite.losses import MNLLLoss, log1pMSELoss
from tangermeme.predict import predict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chrombpnet.adapters import (  # noqa: E402
    ADAPTER_PRESETS,
    adapter_parameters,
    count_adapter_parameters,
    inject_adapters,
    save_adapter_bundle,
)
from finetune_da import (  # noqa: E402
    CHROM_SPLITS_PATH,
    PSEUDOBULKS,
    MixedFinetuneDataset,
    extract_split,
    load_ct_peaks,
    load_da_peaks,
)


def _target_choices():
    return sorted(set(ADAPTER_PRESETS) | {"iconv", "rconvs", "fconv", "linear"})


def _adapter_manifest(args, base_model_path, adapter_config, trainable_params):
    return {
        "format_version": 1,
        "pseudobulk": args.pseudobulk,
        "fold": args.fold,
        "base_model_path": str(base_model_path),
        "sweep_id": args.sweep_id,
        "adapter_config": {
            "targets": adapter_config["targets"],
            "conv_rank": adapter_config["conv_rank"],
            "linear_rank": adapter_config["linear_rank"],
            "conv_scale": adapter_config["conv_scale"],
            "linear_scale": adapter_config["linear_scale"],
            "dropout": adapter_config["dropout"],
        },
        "layer_specs": adapter_config["layer_specs"],
        "trainable_params": trainable_params,
        "training": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "jitter": args.jitter,
            "da_fraction": args.da_fraction,
        },
    }


def train_adapters(
    model,
    train_loader,
    optimizer,
    X_valid,
    y_valid,
    manifest,
    adapter_path,
    log_path,
    max_epochs=50,
    batch_size=64,
    early_stopping=5,
    device="cuda",
    alpha=100,
    verbose=False,
):
    """Train adapter parameters while keeping base ChromBPNet weights frozen."""
    best_loss = float("inf")
    early_stop_count = 0
    log_rows = []

    for epoch in range(max_epochs):
        model.train()
        model.bias.eval()
        t0 = time.time()
        for X, y in train_loader:
            X = X.to(device).float()
            y = y.to(device)

            optimizer.zero_grad()
            y_profile_logits, y_counts = model(X)
            y_profile = F.log_softmax(y_profile_logits.squeeze(1), dim=-1)

            profile_loss = MNLLLoss(y_profile, y).mean()
            count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()
            loss = profile_loss + alpha * count_loss
            loss.backward()
            optimizer.step()
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            model.eval()
            y_profile_val, y_counts_val = predict(
                model, X_valid, batch_size=batch_size, device=device
            )
            y_profile_val = F.log_softmax(y_profile_val.squeeze(1), dim=-1)
            val_profile_loss = MNLLLoss(y_profile_val, y_valid).mean()
            val_count_loss = log1pMSELoss(
                y_counts_val, y_valid.sum(dim=-1).reshape(-1, 1)
            ).mean()
            val_loss = val_profile_loss + alpha * val_count_loss
        val_time = time.time() - t0

        saved = val_loss < best_loss
        log_rows.append(
            {
                "epoch": epoch,
                "train_time_s": round(train_time, 2),
                "val_time_s": round(val_time, 2),
                "train_profile_loss": profile_loss.item(),
                "train_count_loss": count_loss.item(),
                "val_profile_loss": val_profile_loss.item(),
                "val_count_loss": val_count_loss.item(),
                "val_loss": val_loss.item(),
                "saved": saved,
            }
        )

        if verbose:
            print(
                f"    epoch {epoch}: "
                f"train profile={profile_loss.item():.4f} "
                f"count={count_loss.item():.4f} | "
                f"val={val_loss.item():.4f}" + (" *" if saved else "")
            )

        if saved:
            save_adapter_bundle(adapter_path, model, manifest)
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stopping is not None and early_stop_count >= early_stopping:
            break

    pd.DataFrame(log_rows).to_csv(log_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--pseudobulk",
        required=True,
        choices=PSEUDOBULKS,
        help="Cell type to finetune",
    )
    parser.add_argument("-f", "--fold", type=int, required=True, help="Fold index")
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="Name for this adapter run under models/{pseudobulk}/adapters/",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["heads_only"],
        choices=_target_choices(),
        help="Adapter layer targets or presets (default: heads_only)",
    )
    parser.add_argument("--conv-rank", type=int, default=4)
    parser.add_argument("--linear-rank", type=int, default=4)
    parser.add_argument(
        "--conv-alpha",
        type=float,
        default=1.0,
        help="Conv adapter scale alpha; actual scale is alpha / conv_rank",
    )
    parser.add_argument(
        "--linear-alpha",
        type=float,
        default=1.0,
        help="Linear adapter scale alpha; actual scale is alpha / linear_rank",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=100)
    parser.add_argument("--jitter", type=int, default=200)
    parser.add_argument("--da-fraction", type=float, default=0.5)
    parser.add_argument(
        "--materialize-model",
        action="store_true",
        help="Also save a full .torch model with adapters injected",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    ps = args.pseudobulk
    fold = args.fold
    bw_path = str(REPO_ROOT / "data" / "bigwigs" / f"{ps}.bw")
    if not Path(bw_path).exists():
        print(f"Error: bigwig not found: {bw_path}", file=sys.stderr)
        sys.exit(1)

    with open(CHROM_SPLITS_PATH) as f:
        chrom_splits = json.load(f)
    if str(fold) not in chrom_splits:
        print(f"Error: fold {fold} not found in {CHROM_SPLITS_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading DA peaks...")
    da_peaks = load_da_peaks()
    print(f"  {len(da_peaks)} unique DA peaks (tau + pairwise)")

    print(f"Loading cell-type peaks for {ps}...")
    ct_peaks = load_ct_peaks(ps)
    print(f"  {len(ct_peaks)} cell-type peaks")

    out_dir = REPO_ROOT / "models" / ps / "adapters" / args.sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / f"{ps}_f{fold}.adapter.pt"
    log_path = out_dir / f"{ps}_f{fold}.log.tsv"
    config_path = out_dir / f"{ps}_f{fold}.config.json"

    all_chroms = sorted(
        set(da_peaks["chrom"].tolist()) | set(ct_peaks["chrom"].tolist())
    )
    test_chroms = chrom_splits[str(fold)]
    train_chroms = [c for c in all_chroms if c not in test_chroms]
    print(f"\nFold {fold}")
    print(f"  Train chroms: {train_chroms}")
    print(f"  Valid chroms: {test_chroms}")

    base_model_path = REPO_ROOT / "models" / ps / "torch" / f"{ps}_f{fold}.torch"
    print(f"  Loading base model from {base_model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(base_model_path, weights_only=False, map_location=device)
    for parameter in model.parameters():
        parameter.requires_grad = False

    conv_scale = args.conv_alpha / args.conv_rank
    linear_scale = args.linear_alpha / args.linear_rank
    adapter_config = inject_adapters(
        model.accessibility,
        targets=args.targets,
        conv_rank=args.conv_rank,
        linear_rank=args.linear_rank,
        conv_scale=conv_scale,
        linear_scale=linear_scale,
        dropout=args.dropout,
    )
    model.to(device)
    trainable_params = count_adapter_parameters(model.accessibility)
    if trainable_params == 0:
        print("Error: no adapter parameters were injected.", file=sys.stderr)
        sys.exit(1)
    manifest = _adapter_manifest(args, base_model_path, adapter_config, trainable_params)
    with open(config_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"  Injected adapters: {adapter_config['injected']}")
    print(f"  Trainable adapter parameters: {trainable_params:,}")

    nonpeak_path = REPO_ROOT / "data" / "nonpeaks" / f"{ps}_fold_{fold}_negatives.bed"
    if not nonpeak_path.exists():
        print(f"Error: nonpeaks not found: {nonpeak_path}", file=sys.stderr)
        sys.exit(1)
    nonpeaks = pd.read_csv(
        nonpeak_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )

    print("  Extracting training data...")
    X_da_tr, y_da_tr, _ = extract_split(
        da_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    X_ct_tr, y_ct_tr, _ = extract_split(
        ct_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    X_np_tr, y_np_tr, _ = extract_split(
        nonpeaks, bw_path, train_chroms, jitter=0, verbose=args.verbose
    )
    if X_da_tr is None or X_ct_tr is None or X_np_tr is None:
        print("Error: missing training loci after chromosome split.", file=sys.stderr)
        sys.exit(1)

    print(f"    {len(X_da_tr)} DA training peaks")
    print(f"    {len(X_ct_tr)} cell-type training peaks")
    print(f"    {len(X_np_tr)} nonpeaks")

    train_dataset = MixedFinetuneDataset(
        X_da_tr,
        y_da_tr,
        X_ct_tr,
        y_ct_tr,
        X_np_tr,
        y_np_tr,
        jitter=args.jitter,
        da_fraction=args.da_fraction,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    print("  Extracting validation data...")
    X_valid, y_valid, _ = extract_split(
        da_peaks, bw_path, test_chroms, jitter=0, verbose=args.verbose
    )
    if X_valid is None:
        print("Error: no validation DA peaks on test chroms.", file=sys.stderr)
        sys.exit(1)
    print(f"    {len(X_valid)} validation DA peaks")

    optimizer = torch.optim.Adam(
        adapter_parameters(model.accessibility),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"  Finetuning adapters (lr={args.lr}, max_epochs={args.epochs}, "
        f"patience={args.patience})..."
    )
    train_adapters(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        manifest=manifest,
        adapter_path=adapter_path,
        log_path=log_path,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=args.patience,
        device=device,
        alpha=args.alpha,
        verbose=args.verbose,
    )

    if args.materialize_model:
        materialized_dir = REPO_ROOT / "models" / ps / "adapter_models" / args.sweep_id
        materialized_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, materialized_dir / f"{ps}_f{fold}.torch")

    print(f"\nAdapter bundle saved to {adapter_path}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
