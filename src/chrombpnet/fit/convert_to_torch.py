#!/usr/bin/env python3
"""Convert trained ChromBPNet TensorFlow models to torch.

Examples:
    python src/chrombpnet/fit/convert_to_torch.py
    python src/chrombpnet/fit/convert_to_torch.py --dataset fine
    python src/chrombpnet/fit/convert_to_torch.py --dataset coarse -p xylem
"""

import argparse
import sys
from pathlib import Path

import torch
import tqdm
from bpnetlite.bpnet import BPNet
from bpnetlite.chrombpnet import ChromBPNet

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from chrombpnet.datasets import add_dataset_arg, get_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_dataset_arg(parser)
    parser.add_argument(
        "-p",
        "--pseudobulk",
        dest="pseudobulks",
        action="append",
        default=None,
        help="Pseudobulk to convert; repeatable. Defaults to all pseudobulks in --dataset.",
    )
    parser.add_argument(
        "-f",
        "--fold",
        dest="folds",
        type=int,
        action="append",
        choices=range(5),
        default=None,
        help="Fold to convert; repeatable. Defaults to all folds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing torch outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = get_dataset(REPO_ROOT, args.dataset)
    pseudobulks = args.pseudobulks or dataset.pseudobulks()
    folds = args.folds or list(range(5))

    for pseudobulk in pseudobulks:
        dataset.validate_pseudobulk(pseudobulk)
        out_dir = dataset.model_dir(pseudobulk) / "torch"
        out_dir.mkdir(exist_ok=True, parents=True)
        print(f"\nConverting {args.dataset}/{pseudobulk} -> {out_dir}")

        for fold in tqdm.tqdm(folds, desc=pseudobulk):
            out_paths = [
                out_dir / f"{pseudobulk}_f{fold}.torch",
                out_dir / f"{pseudobulk}_f{fold}.bias.torch",
                out_dir / f"{pseudobulk}_f{fold}.accessibility.torch",
            ]
            if not args.overwrite and all(path.exists() for path in out_paths):
                continue

            model_dir = dataset.model_dir(pseudobulk) / f"acc_f{fold}" / "models"
            bias_path = model_dir / "bias_model_scaled.h5"
            accessibility_path = model_dir / "chrombpnet_nobias.h5"
            missing = [path for path in [bias_path, accessibility_path] if not path.exists()]
            if missing:
                missing_txt = "\n".join(str(path) for path in missing)
                raise FileNotFoundError(
                    f"Missing TensorFlow model input(s) for {args.dataset}/{pseudobulk} fold {fold}:\n"
                    f"{missing_txt}"
                )

            bias = BPNet.from_chrombpnet(bias_path)
            accessibility = BPNet.from_chrombpnet(accessibility_path)
            chrombpnet = ChromBPNet(
                bias=bias, accessibility=accessibility, name="chrombpnet"
            )
            torch.save(chrombpnet, out_paths[0])
            torch.save(bias, out_paths[1])
            torch.save(accessibility, out_paths[2])


if __name__ == "__main__":
    main()
