"""
Finetune a Cherimoya model on differentially accessible peaks.

Loads the 5-fold trained Cherimoya models for a given cell type and finetunes
either the full model or daicon adapters using a mixed training set:
  - 50% DA peaks: union of da_peaks_tau.bed + da_peaks_pairwise.bed
  - 50% base mix: cell-type–specific peaks (7/8) + nonpeaks (1/8)

This balances sharpening cell-type specificity while preserving coverage of
the original training distribution to avoid catastrophic forgetting.

Requires differential_accessibility.py outputs to exist in the selected dataset:
    data/peaks{_dataset}/differential_accessibility/da_peaks_tau.bed
    data/peaks{_dataset}/differential_accessibility/da_peaks_pairwise.bed

Full finetuned models are saved to:
    models/{dataset/}{pseudobulk}/finetuned/{pseudobulk}_f{fold}.torch

Daicon adapter bundles are saved to:
    models/{dataset/}{pseudobulk}/adapters/{adapter_run_id}/{pseudobulk}_f{fold}.adapter.pt

Usage:
    uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py -p mesophyll -f 0
    uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py --dataset coarse -p mesophyll -f 0
    uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py -p mesophyll -f 0 --mode adapter --da-fraction 0.8
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
from cherimoya import Cherimoya
from daicon import (  # noqa: E402
    Cherimoya_PRESETS,
    Cherimoya_TARGETS,
    adapter_parameters,
    count_adapter_parameters,
    inject_adapters,
    save_adapter_bundle,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

PSEUDOBULKS = ["bundle_sheath", "epidermis", "guard", "mesophyll", "phloem", "xylem"]
DATASETS = ("base", "fine", "coarse")
IN_WINDOW = 2114
OUT_WINDOW = 1000
# Cherimoya has no single "all" preset (unlike BPNet's iconv+rconvs+fconv+
# linear); all_postblock (blocks + heads) is the broadest available.
DEFAULT_ADAPTER_TARGETS = "all_postblock"
MNLLLoss = None
log1pMSELoss = None
extract_loci = None
predict = None

def dataset_suffix(dataset):
    return "" if dataset == "base" else f"_{dataset}"


def data_subdir(species, name, dataset):
    return species_config.data_dir(species) / f"{name}{dataset_suffix(dataset)}"


def model_root(species, dataset):
    root = species_config.models_dir(species)
    return root if dataset == "base" else root / dataset


def da_dir(species, dataset):
    return data_subdir(species, "peaks", dataset) / "differential_accessibility"


def da_fraction_tag(da_fraction):
    return str(da_fraction).replace(".", "p")


def fine_tag(da_fraction):
    return "finetuned" if da_fraction == 0.5 else f"finetuned_{da_fraction:g}"


def default_adapter_run_id(da_fraction):
    return f"daicon_default_da{da_fraction_tag(da_fraction)}"


def _target_choices():
    return sorted(set(Cherimoya_PRESETS) | set(Cherimoya_TARGETS))


def _load_training_dependencies():
    global MNLLLoss, log1pMSELoss, extract_loci, predict
    if MNLLLoss is not None:
        return

    from bpnetlite.losses import MNLLLoss as _MNLLLoss
    from bpnetlite.losses import log1pMSELoss as _log1pMSELoss
    from tangermeme.io import extract_loci as _extract_loci
    from tangermeme.predict import predict as _predict

    MNLLLoss = _MNLLLoss
    log1pMSELoss = _log1pMSELoss
    extract_loci = _extract_loci
    predict = _predict


class MixedFinetuneDataset(torch.utils.data.Dataset):
    """Mixed training dataset for DA finetuning.

    Each sample is drawn from one of two groups with equal probability:
      - DA peaks (tau-specific + pairwise-DA union): jitter + RC augmentation
      - Base mix: 7/8 cell-type–specific peaks (jitter + RC) + 1/8 nonpeaks (RC only)

    All peak tensors must be pre-extracted with (IN_WINDOW + 2*jitter) /
    (OUT_WINDOW + 2*jitter) padding. Nonpeak tensors use IN_WINDOW / OUT_WINDOW
    directly (no jitter padding).
    """

    def __init__(
        self,
        X_da,
        y_da,
        X_peaks,
        y_peaks,
        X_nonpeaks,
        y_nonpeaks,
        jitter,
        da_fraction=0.5,
    ):
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


def load_da_peaks(species, dataset):
    """Load and deduplicate DA peaks from tau and pairwise BED files."""
    frames = []
    for path in (
        da_dir(species, dataset) / "da_peaks_tau.bed",
        da_dir(species, dataset) / "da_peaks_pairwise.bed",
    ):
        if not path.exists():
            print(
                f"Error: {path} not found. "
                f"Run differential_accessibility.py --dataset {dataset} first.",
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


def load_ct_peaks(species, ps, dataset):
    """Load cell-type–specific peaks as a chrom/start/end DataFrame."""
    path = data_subdir(species, "peaks", dataset) / f"{ps}_peaks.narrowPeak"
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


def extract_split(fasta, loci_df, bw_path, chroms, jitter, verbose):
    """Extract sequences and signal for loci on the given chromosomes.

    Parameters
    ----------
    fasta : str, path to reference genome FASTA
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
    _load_training_dependencies()
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
            sequences=fasta,
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


def _losses(model, X, y, alpha):
    y_profile_logits, y_counts = model(X)
    y_profile = F.log_softmax(y_profile_logits.squeeze(1), dim=-1)
    profile_loss = MNLLLoss(y_profile, y).mean()
    count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()
    loss = profile_loss + alpha * count_loss
    return loss, profile_loss, count_loss


def train_finetune_model(
    model,
    train_loader,
    optimizer,
    X_valid,
    y_valid,
    save_best,
    log_path,
    train_module=None,
    max_epochs=50,
    batch_size=64,
    early_stopping=5,
    device="cuda",
    alpha=100,
    verbose=False,
):
    """Training loop shared by full-model and daicon adapter finetuning.

    Uses model.forward() directly, matching model.fit()'s own call
    convention -- Cherimoya has no bias/accessibility submodel split to
    keep separately frozen or add profiles from.
    """
    best_loss = float("inf")
    early_stop_count = 0
    log_rows = []
    train_module = model if train_module is None else train_module

    _load_training_dependencies()

    for epoch in range(max_epochs):
        train_module.train()
        t0 = time.time()
        for X, y in train_loader:
            X = X.to(device).float()
            y = y.to(device)

            optimizer.zero_grad()
            loss, profile_loss, count_loss = _losses(model, X, y, alpha)
            loss.backward()
            optimizer.step()
        train_time = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            train_module.eval()
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
            save_best()
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stopping is not None and early_stop_count >= early_stopping:
            break

    pd.DataFrame(log_rows).to_csv(log_path, sep="\t", index=False)
    if verbose:
        print(f"    Training log saved to {log_path}")


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
    """Backward-compatible full-model finetuning helper."""

    def save_best():
        model.save(f"{model.name}.torch")

    train_finetune_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        X_valid=X_valid,
        y_valid=y_valid,
        save_best=save_best,
        log_path=f"{model.name}.log",
        train_module=model,
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        device=device,
        alpha=alpha,
        verbose=verbose,
    )


def _adapter_manifest(args, base_model_path, adapter_config, trainable_params):
    return {
        "format_version": 1,
        "dataset": args.dataset,
        "pseudobulk": args.pseudobulk,
        "fold": args.fold,
        "base_model_path": str(base_model_path),
        "adapter_run_id": args.adapter_run_id,
        "sweep_id": args.adapter_run_id,
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
            "mode": args.mode,
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


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "-p",
        "--pseudobulk",
        required=True,
        help="Cell type to finetune",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="base",
        help="Dataset namespace to finetune (default: base).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "adapter"],
        default="full",
        help="Finetuning mode: full accessibility model or daicon adapters (default: full)",
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
        help="Max random jitter in bp for training augmentation (default: 200)",
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
    parser.add_argument(
        "--adapter-run-id",
        default=None,
        help="Adapter output run ID under models/{pseudobulk}/adapters/ (adapter mode only)",
    )
    parser.add_argument(
        "--adapter-targets",
        nargs="+",
        default=[DEFAULT_ADAPTER_TARGETS],
        choices=_target_choices(),
        help="Daicon adapter targets or BPNet presets (default: all)",
    )
    parser.add_argument(
        "--conv-rank",
        type=int,
        default=16,
        help="Daicon Conv1d adapter rank (default: 16)",
    )
    parser.add_argument(
        "--linear-rank",
        type=int,
        default=4,
        help="Daicon Linear adapter rank (default: 4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Daicon adapter dropout (default: 0.0)",
    )
    parser.add_argument(
        "--conv-scale",
        type=float,
        default=None,
        help="Daicon Conv1d adapter scale (default: daicon alpha 8 / conv_rank)",
    )
    parser.add_argument(
        "--linear-scale",
        type=float,
        default=None,
        help="Daicon Linear adapter scale (default: daicon alpha 8 / linear_rank)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay for adapter mode (default: 0.0)",
    )
    parser.add_argument(
        "--materialize-model",
        action="store_true",
        help="Also save a full .torch model with adapters injected (adapter mode only)",
    )
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    species = species_config.resolve_species(args.species)
    pseudobulks = species_config.read_pseudobulks(species, args.dataset)
    if args.pseudobulk not in pseudobulks:
        parser.error(
            f"unknown pseudobulk '{args.pseudobulk}' for species '{species}', "
            f"dataset '{args.dataset}'. Choose one of: {', '.join(pseudobulks)}"
        )
    if args.adapter_run_id is None:
        args.adapter_run_id = default_adapter_run_id(args.da_fraction)
    return args


def main(argv=None):
    args = parse_args(argv)
    species = species_config.resolve_species(args.species)

    ps = args.pseudobulk
    fold = args.fold
    root = model_root(species, args.dataset)
    bw_path = str(data_subdir(species, "bigwigs", args.dataset) / f"{ps}.bw")
    if not Path(bw_path).exists():
        print(f"Error: bigwig not found: {bw_path}", file=sys.stderr)
        sys.exit(1)

    fasta = str(species_config.genome_fasta(species))
    chrom_splits_path = species_config.config_dir(species) / "chrom_splits.json"
    with open(chrom_splits_path) as f:
        chrom_splits = json.load(f)

    if str(fold) not in chrom_splits:
        print(f"Error: fold {fold} not found in {chrom_splits_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading DA peaks...")
    da_peaks = load_da_peaks(species, args.dataset)
    print(f"  {len(da_peaks)} unique DA peaks (tau + pairwise)")

    print(f"Loading cell-type peaks for {ps}...")
    ct_peaks = load_ct_peaks(species, ps, args.dataset)
    print(f"  {len(ct_peaks)} cell-type peaks")

    if args.mode == "adapter":
        out_dir = root / ps / "adapters" / args.adapter_run_id
    else:
        out_dir = root / ps / fine_tag(args.da_fraction)
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
    model_path = root / ps / f"{ps}_f{fold}.torch"
    print(f"  Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Cherimoya.load(model_path, device=device, compile=False)

    adapter_path = out_dir / f"{ps}_f{fold}.adapter.pt"
    adapter_log_path = out_dir / f"{ps}_f{fold}.log.tsv"
    adapter_config_path = out_dir / f"{ps}_f{fold}.config.json"

    if args.mode == "adapter":
        for param in model.parameters():
            param.requires_grad = False

        adapter_config = inject_adapters(
            model,
            targets=args.adapter_targets,
            conv_rank=args.conv_rank,
            linear_rank=args.linear_rank,
            conv_scale=args.conv_scale,
            linear_scale=args.linear_scale,
            dropout=args.dropout,
            architecture="cherimoya",
        )
        model.to(device)
        trainable_params = count_adapter_parameters(model)
        if trainable_params == 0:
            print("Error: no adapter parameters were injected.", file=sys.stderr)
            sys.exit(1)
        manifest = _adapter_manifest(args, model_path, adapter_config, trainable_params)
        with open(adapter_config_path, "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"  Injected adapters: {adapter_config['injected']}")
        print(f"  Trainable adapter parameters: {trainable_params:,}")
    else:
        model.name = str(out_dir / f"{ps}_f{fold}")
        for param in model.parameters():
            param.requires_grad = True

    # --- Load nonpeaks ---
    nonpeak_path = (
        data_subdir(species, "nonpeaks", args.dataset) / f"{ps}_fold_{fold}_negatives.bed"
    )
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
        fasta, da_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    X_ct_tr, y_ct_tr, _ = extract_split(
        fasta, ct_peaks, bw_path, train_chroms, jitter=args.jitter, verbose=args.verbose
    )
    # Nonpeaks: no jitter — extracted at IN_WINDOW / OUT_WINDOW directly
    X_np_tr, y_np_tr, _ = extract_split(
        fasta, nonpeaks, bw_path, train_chroms, jitter=0, verbose=args.verbose
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
        fasta, da_peaks, bw_path, test_chroms, jitter=0, verbose=args.verbose
    )

    if X_valid is None:
        print("Error: no validation DA peaks on test chroms.", file=sys.stderr)
        sys.exit(1)

    print(f"    {len(X_valid)} validation DA peaks")

    # --- Finetune ---
    if args.mode == "adapter":
        optimizer = torch.optim.Adam(
            adapter_parameters(model),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        def save_best():
            save_adapter_bundle(adapter_path, model, manifest, adapter_root=None)

        print(
            f"  Finetuning daicon adapters (lr={args.lr}, max_epochs={args.epochs}, "
            f"patience={args.patience})..."
        )
        train_finetune_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            X_valid=X_valid,
            y_valid=y_valid,
            save_best=save_best,
            log_path=adapter_log_path,
            train_module=model,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping=args.patience,
            device=device,
            alpha=args.alpha,
            verbose=args.verbose,
        )
        if args.materialize_model:
            materialized_dir = (
                root / ps / "adapter_models" / args.adapter_run_id
            )
            materialized_dir.mkdir(parents=True, exist_ok=True)
            model.save(materialized_dir / f"{ps}_f{fold}.torch")
        print(f"\nAdapter bundle saved to {adapter_path}")
        print(f"Training log saved to {adapter_log_path}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            alpha=args.alpha,
            verbose=args.verbose,
        )
        print(f"\nFinetuned model saved to {out_dir / f'{ps}_f{fold}.torch'}")


if __name__ == "__main__":
    main()
