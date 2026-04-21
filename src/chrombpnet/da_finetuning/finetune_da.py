"""
Finetune ChromBPNet accessibility model on differentially accessible peaks.

Loads the 5-fold trained ChromBPNet models for a given cell type and finetunes
the accessibility submodel (bias frozen) using a mixed training set:
  - 50% DA peaks: union of da_peaks_tau.bed + da_peaks_pairwise.bed
  - 50% base mix: cell-type–specific peaks (7/8) + nonpeaks (1/8)

This balances sharpening cell-type specificity while preserving coverage of
the original training distribution to avoid catastrophic forgetting.

Requires differential_accessibility.py outputs to exist:
    data/peaks/differential_accessibility/da_peaks_tau.bed
    data/peaks/differential_accessibility/da_peaks_pairwise.bed

Finetuned models are saved to:
    models/{pseudobulk}/finetuned/{pseudobulk}_f{fold}.torch
    models/{pseudobulk}/finetuned/{pseudobulk}_f{fold}.accessibility.torch

Usage:
    python src/chrombpnet/da_finetuning/finetune_da.py -p mesophyll -f 0
    python src/chrombpnet/da_finetuning/finetune_da.py -p mesophyll -f 0 --lr 5e-5 --epochs 30
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from bpnetlite.losses import MNLLLoss, log1pMSELoss
from tangermeme.io import extract_loci
from tangermeme.predict import predict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
FASTA = str(REPO_ROOT / "data" / "Sbicolor_730_v5.0.fa")
CHROM_SPLITS_PATH = REPO_ROOT / "configs" / "chrom_splits.json"
IN_WINDOW = 2114
OUT_WINDOW = 1000
DA_TAU_BED = (
    REPO_ROOT / "data" / "peaks" / "differential_accessibility" / "da_peaks_tau.bed"
)
DA_PW_BED = (
    REPO_ROOT
    / "data"
    / "peaks"
    / "differential_accessibility"
    / "da_peaks_pairwise.bed"
)


class MixedFinetuneDataset(torch.utils.data.Dataset):
    """Mixed training dataset for DA finetuning.

    Each sample is drawn from one of two groups with equal probability:
      - DA peaks (tau-specific + pairwise-DA union): jitter + RC augmentation
      - Base mix: 7/8 cell-type–specific peaks (jitter + RC) + 1/8 nonpeaks (RC only)

    All peak tensors must be pre-extracted with (IN_WINDOW + 2*jitter) /
    (OUT_WINDOW + 2*jitter) padding. Nonpeak tensors use IN_WINDOW / OUT_WINDOW
    directly (no jitter padding).
    """

    def __init__(self, X_da, y_da, X_peaks, y_peaks, X_nonpeaks, y_nonpeaks, jitter, da_fraction=0.5):
        """
        Parameters
        ----------
        X_da : torch.Tensor, shape (n_da, 4, IN_WINDOW + 2*jitter)
        y_da : torch.Tensor, shape (n_da, OUT_WINDOW + 2*jitter)
        X_peaks : torch.Tensor, shape (n_peaks, 4, IN_WINDOW + 2*jitter)
        y_peaks : torch.Tensor, shape (n_peaks, OUT_WINDOW + 2*jitter)
        X_nonpeaks : torch.Tensor, shape (n_nonpeaks, 4, IN_WINDOW)
        y_nonpeaks : torch.Tensor, shape (n_nonpeaks, OUT_WINDOW)
        jitter : int
            Maximum jitter in bp applied to peaks/DA peaks.
        """
        self.X_da = X_da
        self.y_da = y_da
        self.X_peaks = X_peaks
        self.y_peaks = y_peaks
        self.X_nonpeaks = X_nonpeaks
        self.y_nonpeaks = y_nonpeaks
        self.jitter = jitter
        self.n_da = len(X_da)
        self.n_peaks = len(X_peaks)
        self.n_nonpeaks = len(X_nonpeaks)
        self.da_fraction = da_fraction
        # Epoch length: n_da DA samples (= da_fraction of epoch) + proportional base samples.
        # With da_fraction=0.5: epoch = 2*n_da; with da_fraction=0.8: epoch = 5/4*n_da.
        self.epoch_len = int(self.n_da / da_fraction)

    def __len__(self):
        return self.epoch_len

    def _jitter_crop(self, X, y):
        """Apply a random jitter offset and crop to IN_WINDOW / OUT_WINDOW."""
        if self.jitter > 0:
            offset = torch.randint(0, 2 * self.jitter + 1, (1,)).item()
        else:
            offset = 0
        X_crop = X[:, offset : offset + IN_WINDOW]
        y_crop = y[offset : offset + OUT_WINDOW]
        return X_crop, y_crop

    def _rc(self, X, y):
        """Apply reverse-complement augmentation with 50% probability."""
        if torch.rand(1).item() < 0.5:
            # Flip along sequence axis, then reorder channels: ACGT → TGCA
            X = torch.flip(X, dims=[-1])[[3, 2, 1, 0], :]
            y = torch.flip(y, dims=[-1])
        return X, y

    def __getitem__(self, idx):
        if idx < self.n_da:
            # --- DA peaks: use idx directly so each peak is seen exactly once ---
            X, y = self._jitter_crop(self.X_da[idx], self.y_da[idx])
        else:
            # --- Base mix: 7/8 cell-type peaks, 1/8 nonpeaks ---
            if torch.rand(1).item() < 7 / 8:
                i = torch.randint(0, self.n_peaks, (1,)).item()
                X, y = self._jitter_crop(self.X_peaks[i], self.y_peaks[i])
            else:
                i = torch.randint(0, self.n_nonpeaks, (1,)).item()
                # Nonpeaks already at IN_WINDOW / OUT_WINDOW — no jitter crop needed
                X = self.X_nonpeaks[i]
                y = self.y_nonpeaks[i]

        return self._rc(X, y)


def load_da_peaks():
    """Load and deduplicate DA peaks from tau and pairwise BED files."""
    frames = []
    for path in (DA_TAU_BED, DA_PW_BED):
        if not path.exists():
            print(
                f"Error: {path} not found. Run differential_accessibility.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
        )
        frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["chrom", "start", "end"])
        .sort_values(["chrom", "start"])
        .reset_index(drop=True)
    )


def load_ct_peaks(ps):
    """Load cell-type–specific peaks as a chrom/start/end DataFrame."""
    path = REPO_ROOT / "data" / "peaks" / f"{ps}_peaks.narrowPeak"
    if not path.exists():
        print(f"Error: peaks file not found: {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"]
    )
    return df.sort_values(["chrom", "start"]).reset_index(drop=True)


def peaks_to_narrowpeak(df, tmp_file):
    """Write a chrom/start/end DataFrame to a 10-column narrowPeak temp file."""
    np_df = df[["chrom", "start", "end"]].copy()
    np_df["name"] = "."
    np_df["score"] = 0
    np_df["strand"] = "."
    np_df["signalValue"] = -1
    np_df["pValue"] = -1
    np_df["qValue"] = -1
    np_df["peak"] = ((df["end"] - df["start"]) // 2).astype(int)
    np_df.to_csv(tmp_file, sep="\t", header=False, index=False)


def extract_split(loci_df, bw_path, chroms, jitter, verbose):
    """Extract sequences and signal for loci on the given chromosomes.

    Parameters
    ----------
    loci_df : DataFrame with columns [chrom, start, end]
    bw_path : str, path to bigwig signal file
    chroms : list of str, chromosome names to include
    jitter : int, extra padding on each side (0 for nonpeaks)
    verbose : bool

    Returns
    -------
    X : torch.Tensor, shape (n_valid, 4, IN_WINDOW + 2*jitter)
    y : torch.Tensor, shape (n_valid, OUT_WINDOW + 2*jitter)
    valid_indices : list[int], positions in loci_df that were successfully extracted
    """
    split_df = loci_df[loci_df["chrom"].isin(chroms)].reset_index(drop=True)
    original_indices = loci_df.index[loci_df["chrom"].isin(chroms)].tolist()

    if len(split_df) == 0:
        return None, None, []

    padded_in = IN_WINDOW + 2 * jitter
    padded_out = OUT_WINDOW + 2 * jitter

    with tempfile.NamedTemporaryFile(suffix=".narrowPeak", mode="w", delete=False) as f:
        peaks_to_narrowpeak(split_df, f)
        tmp_path = f.name

    try:
        X, y, mask = extract_loci(
            loci=tmp_path,
            sequences=FASTA,
            signals=[bw_path],
            in_window=padded_in,
            out_window=padded_out,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
            verbose=verbose,
            return_mask=True,
        )
    finally:
        Path(tmp_path).unlink()

    mask_list = mask.tolist()
    valid_indices = [original_indices[i] for i, m in enumerate(mask_list) if m]

    # y: (n_valid, 1, padded_out) → (n_valid, padded_out); abs for strand-agnostic signal
    y = y[:, 0, :].abs()

    return X, y, valid_indices


def finetune(
    model,
    train_loader,
    optimizer,
    X_valid,
    y_valid,
    max_epochs=50,
    batch_size=64,
    early_stopping=5,
    device="cuda",
    alpha=100,
    verbose=False,
):
    """Training loop for ChromBPNet finetuning.

    Uses model.forward() (ChromBPNet) for the forward pass so that the bias
    profile is correctly cropped before being added to the accessibility profile.
    bpnetlite's built-in fit() calls the two submodels separately and adds their
    profiles directly, which fails when they have different output widths.

    Validates once per epoch. Saves best model (by validation loss) to
    {model.name}.torch and {model.name}.accessibility.torch, and the final
    model to {model.name}.final.torch.
    """
    best_loss = float("inf")
    early_stop_count = 0
    log_rows = []

    model.bias.eval()

    for epoch in range(max_epochs):
        model.accessibility.train()
        t0 = time.time()
        for X, y in train_loader:
            X = X.to(device).float()
            y = y.to(device)

            optimizer.zero_grad()

            # ChromBPNet.forward() crops bias profile to match accessibility output
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
            model.accessibility.eval()
            # predict() from tangermeme batches X_valid and calls model.forward(),
            # returning CPU tensors regardless of device.
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
            torch.save(model, f"{model.name}.torch")
            torch.save(model.accessibility, f"{model.name}.accessibility.torch")
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stopping is not None and early_stop_count >= early_stopping:
            break

    log_path = f"{model.name}.log"
    pd.DataFrame(log_rows).to_csv(log_path, sep="\t", index=False)
    if verbose:
        print(f"    Training log saved to {log_path}")


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
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Max training epochs (default: 100)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=100,
        help="Balancing term between profile and count loss (default: 100)",
    )
    parser.add_argument(
        "--jitter",
        type=int,
        default=200,
        help="Max random jitter in bp for training augmentation (default: 500)",
    )
    parser.add_argument(
        "-f", "--fold", type=int, required=True, help="Fold index to finetune (0-based)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--da-fraction",
        type=float,
        default=0.5,
        help="Fraction of each epoch drawn from DA peaks (default: 0.5 = 50:50 split)",
    )
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

    fine_tag = "finetuned" if args.da_fraction == 0.5 else f"finetuned_{args.da_fraction:g}"
    out_dir = REPO_ROOT / "models" / ps / fine_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chroms = sorted(
        set(da_peaks["chrom"].tolist()) | set(ct_peaks["chrom"].tolist())
    )

    test_chroms = chrom_splits[str(fold)]
    train_chroms = [c for c in all_chroms if c not in test_chroms]
    print(f"\nFold {fold}")
    print(f"  Train chroms: {train_chroms}")
    print(f"  Valid chroms: {test_chroms}")

    # --- Load source model ---
    model_path = REPO_ROOT / "models" / ps / "torch" / f"{ps}_f{fold}.torch"
    print(f"  Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.name = str(out_dir / f"{ps}_f{fold}")

    # Bias is already frozen at construction; ensure accessibility is trainable
    for param in model.accessibility.parameters():
        param.requires_grad = True

    # --- Load nonpeaks ---
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

    # --- Extract training data ---
    print("  Extracting training data...")

    X_da_tr, y_da_tr, _ = extract_split(
        da_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    X_ct_tr, y_ct_tr, _ = extract_split(
        ct_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    # Nonpeaks: no jitter — extracted at IN_WINDOW / OUT_WINDOW directly
    X_np_tr, y_np_tr, _ = extract_split(
        nonpeaks, bw_path, train_chroms, jitter=0, verbose=args.verbose
    )

    if X_da_tr is None:
        print("Error: no DA training peaks on train chroms.", file=sys.stderr)
        sys.exit(1)
    if X_ct_tr is None:
        print("Error: no cell-type training peaks on train chroms.", file=sys.stderr)
        sys.exit(1)
    if X_np_tr is None:
        print("Error: no nonpeaks on train chroms.", file=sys.stderr)
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

    # --- Extract validation data (DA peaks on test chroms, no jitter) ---
    print("  Extracting validation data...")
    X_valid, y_valid, _ = extract_split(
        da_peaks, bw_path, test_chroms, jitter=0, verbose=args.verbose
    )

    if X_valid is None:
        print("Error: no validation DA peaks on test chroms.", file=sys.stderr)
        sys.exit(1)

    print(f"    {len(X_valid)} validation DA peaks")

    # --- Finetune ---
    optimizer = torch.optim.Adam(model.accessibility.parameters(), lr=args.lr)
    print(
        f"  Finetuning (lr={args.lr}, max_epochs={args.epochs}, "
        f"patience={args.patience})..."
    )
    finetune(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=args.patience,
        device=device,
        verbose=args.verbose,
    )

    print(f"\nFinetuned model saved to {out_dir / f'{ps}_f{fold}.torch'}")


if __name__ == "__main__":
    main()
