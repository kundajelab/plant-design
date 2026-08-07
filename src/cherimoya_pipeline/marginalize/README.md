# ChromBPNet Marginalization

This directory contains motif marginalization helpers built around PyTorch
ChromBPNet models, tangermeme, and bpnetlite.

Run commands from the repository root with `uv run python`.

## Inputs

```text
models/{dataset/}<pseudobulk>/torch/<pseudobulk>_f<fold>.torch
attributions/chrombpnet/<pseudobulk>_ohe.npz
data/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt
```

The report script takes explicit paths, so it can also be pointed at finetuned
or adapter-materialized models.

## Report

Generate a marginalization report for a sampled set of loci:

```bash
uv run python src/chrombpnet/marginalize/report.py \
    models/mesophyll/torch/mesophyll_f0.torch \
    data/Sbicolor_730_v5.0.fa \
    data/peaks/mesophyll_peaks.narrowPeak \
    data/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt \
    marginalization/mesophyll_f0
```

Useful options:

```bash
uv run python src/chrombpnet/marginalize/report.py \
    models/coarse/mesophyll/torch/mesophyll_f0.torch \
    data/Sbicolor_730_v5.0.fa \
    data/peaks_coarse/mesophyll_peaks.narrowPeak \
    data/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt \
    marginalization/coarse_mesophyll_f0 \
    --n 250 \
    --batch_size 32 \
    --random_state 1234 \
    --verbose
```

## Notes

`marginalize.py` provides the reusable Python functions. `report.py` is the
command-line entrypoint that loads a model, extracts sequences, runs motif
insertions, and writes an HTML report with the generated figures.
