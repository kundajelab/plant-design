# Differential Accessibility Finetuning

These scripts fine-tune converted PyTorch ChromBPNet models on DA-enriched
training sets. Run commands from the repository root with `uv run python`.

## Inputs

Run these steps first for the same dataset:

```bash
uv run python src/data_preprocessing/differential_accessibility.py --dataset coarse
uv run python src/chrombpnet/fit/convert_to_torch.py
```

Required files:

```text
configs/pseudobulks{_dataset}.txt
configs/fold_0.json ... configs/fold_4.json
data/Sbicolor_730_v5.0.fa
data/bigwigs{_dataset}/<pseudobulk>.bw
data/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/nonpeaks{_dataset}/<pseudobulk>_fold_<fold>_negatives.bed
data/peaks{_dataset}/differential_accessibility/da_peaks_tau.bed
data/peaks{_dataset}/differential_accessibility/da_peaks_pairwise.bed
models/{dataset/}<pseudobulk>/torch/<pseudobulk>_f<fold>.torch
```

## Single Runs

Full-model fine-tuning:

```bash
uv run python src/chrombpnet/da_finetuning/finetune_da.py \
    --dataset coarse \
    -p mesophyll \
    -f 0 \
    --mode full
```

Adapter fine-tuning:

```bash
uv run python src/chrombpnet/da_finetuning/finetune_da.py \
    --dataset coarse \
    -p mesophyll \
    -f 0 \
    --mode adapter
```

The standard DA fractions are `0.5` and `0.8`:

```bash
uv run python src/chrombpnet/da_finetuning/finetune_da.py \
    --dataset fine \
    -p bundle_sheath \
    -f 0 \
    --mode adapter \
    --da-fraction 0.8
```

## Standard Launcher

Submit all standard full-model and adapter runs:

```bash
uv run python src/chrombpnet/da_finetuning/launch.py
uv run python src/chrombpnet/da_finetuning/launch.py --dataset coarse
uv run python src/chrombpnet/da_finetuning/launch.py --dataset fine
```

Limit to one run group, pseudobulk, or fold:

```bash
uv run python src/chrombpnet/da_finetuning/launch.py \
    --dataset coarse \
    --run adapter \
    --pseudobulk mesophyll \
    --fold 0
```

Print commands for local scheduling:

```bash
uv run python src/chrombpnet/da_finetuning/launch.py \
    --dataset fine \
    --interactive \
    --rerun-finished
```

Check expected logs without launching:

```bash
uv run python src/chrombpnet/da_finetuning/launch.py \
    --dataset coarse \
    --check-finished
```

## Adapter Sweeps

Run a configured adapter sweep:

```bash
uv run python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --dataset coarse \
    --config configs/adapter_sweeps/smoke.json
```

Print sweep commands:

```bash
uv run python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --dataset coarse \
    --interactive \
    --rerun-finished
```

## Outputs

Base dataset outputs are written below `models/<pseudobulk>/`. Coarse and fine
outputs are written below `models/coarse/<pseudobulk>/` and
`models/fine/<pseudobulk>/`.

```text
finetuned/<pseudobulk>_f<fold>.torch
finetuned_<fraction>/<pseudobulk>_f<fold>.torch
adapters/<run_id>/<pseudobulk>_f<fold>.adapter.pt
adapter_models/<run_id>/<pseudobulk>_f<fold>.torch
```
