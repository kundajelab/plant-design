#!/usr/bin/env python3
"""Collect and print unfinished tf-modisco motif jobs on Cherimoya attributions.

Reads pseudobulks from configs/<species>/pseudobulks.txt and prints one
shell command per (pseudobulk, head) pair via `modisco motifs` whose output
is missing; pipe into `parallel` or run directly.

Jobs are skipped if the modisco h5 output already exists or if the required
attribution/OHE npz files are missing (run attribute/launch.py first).
Run launch_report.py afterwards to generate HTML reports from the h5 files.

Usage:
    python src/cherimoya_pipeline/modisco/launch.py                                    # all pseudobulks, both heads
    python src/cherimoya_pipeline/modisco/launch.py --head count                       # count head only
    python src/cherimoya_pipeline/modisco/launch.py --finetuned                        # finetuned model attributions (50:50 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --finetuned --da-fraction 0.8     # finetuned model attributions (80:20 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --adapter                         # default daicon adapter attributions (50:50 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --adapter --da-fraction 0.8       # default daicon adapter attributions (80:20 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --da-peaks tau                    # modisco on tau-specific DA peaks only
    python src/cherimoya_pipeline/modisco/launch.py --da-peaks specific_union         # modisco on per-cell-type union DA peaks
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attribute"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
from da_peaks import DA_PEAK_CHOICES, da_suffix, read_pseudobulks, resolve_ohe_path
from model_variants import add_model_variant_args, resolve_model_variant

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import species_config

JASPAR_PATH = REPO_ROOT / "data" / "JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt"


def first_array_rows(npz_path):
    with zipfile.ZipFile(npz_path) as zf:
        npy_name = next(name for name in zf.namelist() if name.endswith(".npy"))
        with zf.open(npy_name) as npy:
            version = np.lib.format.read_magic(npy)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(npy)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(npy)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(npy)
    return shape[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    species_config.add_species_arg(parser)
    parser.add_argument(
        "--head",
        type=str,
        action="append",
        choices=["profile", "count"],
        default=None,
        metavar="HEAD",
        help="attribution head(s) to run; repeatable (default: profile count)",
    )
    add_model_variant_args(
        parser,
        finetuned_help="Use attributions from finetuned models instead of base models",
        adapter_help="Use attributions from default daicon adapter bundles for the selected DA fraction",
    )
    parser.add_argument(
        "--da-peaks",
        type=str,
        choices=DA_PEAK_CHOICES,
        default=None,
        help="Run modisco on DA-peaks-only attributions (default: all peaks)",
    )
    # Modisco parameters
    parser.add_argument(
        "-n",
        "--n-seqlets",
        type=int,
        default=1_000_000,
        help="max number of seqlets to use (default: 1_000_000)",
    )
    parser.add_argument(
        "-l",
        "--leiden",
        type=int,
        default=50,
        help="number of leiden clusters to use (default: 50)",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=400,
        help="window to get seqlets from (default: 400)",
    )
    args = parser.parse_args()
    species = species_config.resolve_species(args.species)
    variant = resolve_model_variant(args)
    fine_tag = variant["fine_tag"]
    adapter_id = variant["adapter_id"]

    heads = args.head if args.head is not None else ["profile", "count"]

    if adapter_id is not None:
        attr_subdir = f"cherimoya_adapter_{adapter_id}"
    else:
        attr_subdir = f"cherimoya_{fine_tag}" if args.finetuned else "cherimoya"
    modisco_subdir = attr_subdir  # mirrors attribution directory naming
    suffix = da_suffix(args.da_peaks)

    # Load pseudobulk list
    pseudobulks = read_pseudobulks(species, "base")

    attr_dir = species_config.attributions_dir(species) / attr_subdir
    out_dir = species_config.modisco_dir(species) / modisco_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    printed = 0
    skipped_done = 0
    skipped_no_attr = 0
    skipped_mismatch = 0
    for pseudobulk in pseudobulks:
        # OHE sequences are shared across model variants but must match the loci set.
        ohe_path = resolve_ohe_path(species, pseudobulk, args.da_peaks)
        if not ohe_path.exists():
            skipped_no_attr += len(heads)
            continue

        for head in heads:
            attr_path = attr_dir / f"{pseudobulk}_{head}{suffix}.npz"
            if not attr_path.exists():
                skipped_no_attr += 1
                continue

            if first_array_rows(ohe_path) != first_array_rows(attr_path):
                skipped_mismatch += 1
                print(
                    f"SKIP {pseudobulk}/{head}: OHE and attribution row counts differ "
                    f"({ohe_path} vs {attr_path})",
                    file=sys.stderr,
                )
                continue

            # Skip if modisco output already exists
            out_h5 = out_dir / f"{pseudobulk}_{head}{suffix}.modisco.h5"
            if out_h5.exists():
                skipped_done += 1
                continue

            modisco_motifs_cmd = (
                f"modisco motifs"
                f" -s {ohe_path}"
                f" -a {attr_path}"
                f" -o {out_h5}"
                f" -n {args.n_seqlets} -l {args.leiden} -w {args.window} -v"
            )

            print(modisco_motifs_cmd)
            printed += 1

    total = len(pseudobulks) * len(heads)
    print(
        f"\nPrinted {printed} command(s), skipped {skipped_no_attr} missing attributions, "
        f"skipped {skipped_mismatch} row-count mismatches, "
        f"skipped {skipped_done} already done ({total} total)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
