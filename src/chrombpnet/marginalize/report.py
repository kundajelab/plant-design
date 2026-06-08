# report.py
# Adam He <adamyhe@gmail.com>

"""
Script to generate marginalization reports. Adapted from bpnetlite's
marginalization_report, but makes some slight modifications to fix bugs &
allow for saving vector images.

USAGE:
python report.py model_fname genome_fname bed_fname meme_fname output_dir
"""

import argparse
import os

import marginalize
import numpy as np
import torch
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.io import extract_loci


def main(
    model_fname,
    sequence_fname,
    loci_fname,
    motif_fname,
    output_dir,
    n=100,
    batch_size=16,
    verbose=False,
    random_state=0,
):
    # Load models
    model = torch.load(model_fname, weights_only=False)

    # Load data
    X = extract_loci(
        sequences=sequence_fname,
        loci=loci_fname,
        max_jitter=0,
        n_loci=n,
        verbose=verbose,
        ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
    ).float()

    # Create random sequences to marginalize across
    X_dinuc = torch.stack(
        [
            dinucleotide_shuffle(X[i : i + 1], random_state=random_state, n=1).squeeze()
            for i in range(X.shape[0])
        ],
        axis=0,
    )

    # Create marginalization report
    os.makedirs(output_dir, exist_ok=True)
    results = marginalize.marginalization_report(
        model, motif_fname, X_dinuc, output_dir, batch_size=batch_size, verbose=verbose
    )
    np.savez_compressed(
        os.path.join(output_dir, "marginalization_results.npz"), **results
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model_fname", type=str)
    parser.add_argument("sequence_fname", type=str)
    parser.add_argument("loci_fname", type=str)
    parser.add_argument("motif_fname", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--random_state", type=int, default=None)
    args = parser.parse_args()

    main(**vars(args))
