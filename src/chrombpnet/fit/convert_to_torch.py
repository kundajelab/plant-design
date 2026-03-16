# Convert chrombpnet models to torch

from pathlib import Path

import torch
import tqdm
from bpnetlite.bpnet import BPNet
from bpnetlite.chrombpnet import ChromBPNet

for pseudobulk in [
#    "bundle_sheath",
#    "epidermis",
#    "guard",
#    "mesophyll",
#    "phloem",
    "xylem",
]:
    out_dir = Path(f"../../../models/{pseudobulk}/torch/")
    out_dir.mkdir(exist_ok=True, parents=True)
    print(out_dir)
    for i in tqdm.trange(5):
        model_dir = Path(f"../../../models/{pseudobulk}/acc_f{i}/models/")
        print(model_dir)
        bias = BPNet.from_chrombpnet(model_dir / "bias_model_scaled.h5")
        accessibility = BPNet.from_chrombpnet(model_dir / "chrombpnet_nobias.h5")
        chrombpnet = ChromBPNet(
            bias=bias, accessibility=accessibility, name="chrombpnet"
        )
        torch.save(chrombpnet, out_dir / f"{pseudobulk}_f{i}.torch")
        torch.save(bias, out_dir / f"{pseudobulk}_f{i}.bias.torch")
        torch.save(accessibility, out_dir / f"{pseudobulk}_f{i}.accessibility.torch")
