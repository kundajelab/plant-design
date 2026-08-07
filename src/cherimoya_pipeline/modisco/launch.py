#!/usr/bin/env python3
"""Submit SLURM jobs to run tf-modisco motifs on Cherimoya attributions.

Reads pseudobulks from configs/pseudobulks.txt and submits one sbatch job per
(pseudobulk, head) pair via `modisco motifs`.

Jobs are skipped if the modisco h5 output already exists or if the required
attribution/OHE npz files are missing (run attribute/launch.py first).
Run launch_report.py afterwards to generate HTML reports from the h5 files.

Usage:
    python src/cherimoya_pipeline/modisco/launch.py                                    # all pseudobulks, both heads
    python src/cherimoya_pipeline/modisco/launch.py --dry-run                          # print sbatch scripts without submitting
    python src/cherimoya_pipeline/modisco/launch.py --head count                       # count head only
    python src/cherimoya_pipeline/modisco/launch.py --finetuned                        # finetuned model attributions (50:50 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --finetuned --da-fraction 0.8     # finetuned model attributions (80:20 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --adapter                         # default daicon adapter attributions (50:50 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --adapter --da-fraction 0.8       # default daicon adapter attributions (80:20 DA split)
    python src/cherimoya_pipeline/modisco/launch.py --da-peaks tau                    # modisco on tau-specific DA peaks only
    python src/cherimoya_pipeline/modisco/launch.py --da-peaks specific_union         # modisco on per-cell-type union DA peaks
    python src/cherimoya_pipeline/modisco/launch.py --time 6-23:00:00 --mem 64G
"""

import argparse
import subprocess
import sys
import textwrap
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
        "--dry-run",
        action="store_true",
        help="print sbatch scripts without submitting",
    )
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
    # SLURM resource flags
    parser.add_argument("--partition", type=str, default="normal,akundaje,owners")
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--time", type=str, default="2-00:00:00")
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
    log_dir = species_config.logs_dir(species) / "cherimoya_modisco"
    log_dir.mkdir(parents=True, exist_ok=True)

    submitted = 0
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

            if adapter_id is not None:
                tag = f"_adapter_{adapter_id}"
            else:
                tag = f"_{fine_tag}" if args.finetuned else ""
            job_name = f"modisco{tag}{suffix}_{pseudobulk}_{head}"

            modisco_motifs_cmd = (
                f"modisco motifs"
                f" -s {ohe_path}"
                f" -a {attr_path}"
                f" -o {out_h5}"
                f" -n {args.n_seqlets} -l {args.leiden} -w {args.window} -v"
            )

            sbatch_script = textwrap.dedent(f"""\
                #!/bin/bash -l
                #SBATCH --job-name={job_name}
                #SBATCH --ntasks=1
                #SBATCH --ntasks-per-node=1
                #SBATCH --nodes=1
                #SBATCH --cpus-per-task={args.cpus_per_task}
                #SBATCH --mem={args.mem}
                #SBATCH --partition={args.partition}
                #SBATCH --time={args.time}
                #SBATCH --output={log_dir}/{job_name}.out
                #SBATCH --error={log_dir}/{job_name}.err
                #SBATCH -C NO_GPU

                ml openblas/0.3.28
                ml xsimd/8.1.0
                ml xz/5.8.1
                ml hdf5/1.14.4
                ml arrow/22.0.0
                ml load py-pyarrow/18.1.0_py312
                ml lz4/1.8.0
                ml biology
                ml htslib
                ml ucsc-utils

                mamba activate torch
                export NUMBA_NUM_THREADS={args.cpus_per_task}

                mkdir -p {out_dir}
                cd {out_dir}
                time {modisco_motifs_cmd}
            """)

            if args.dry_run:
                print(f"--- {job_name} ---")
                print(sbatch_script)
                submitted += 1
                continue

            result = subprocess.run(
                ["sbatch"], input=sbatch_script, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"{job_name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(
                    f"ERROR submitting {job_name}: {result.stderr.strip()}",
                    file=sys.stderr,
                )

    action = "Would submit" if args.dry_run else "Submitted"
    total = len(pseudobulks) * len(heads)
    print(
        f"\n{action} {submitted} jobs, skipped {skipped_no_attr} missing attributions, "
        f"skipped {skipped_mismatch} row-count mismatches, "
        f"skipped {skipped_done} already done ({total} total)"
    )


if __name__ == "__main__":
    main()
