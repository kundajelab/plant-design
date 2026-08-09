# Differential Accessibility Finetuning

These scripts fine-tune trained Cherimoya models on DA-enriched
training sets. Run commands from the repository root with `uv run python`.

## Inputs

Run these steps first for the same dataset:

```bash
uv run python src/data_preprocessing/differential_accessibility.py --dataset coarse
uv run python src/cherimoya_pipeline/fit/launch.py --dataset coarse
```

Required files (`--species` defaults to `S.bicolor`):

```text
configs/<species>/pseudobulks{_dataset}.txt
configs/<species>/fold_0.json ... fold_4.json
data/<species>/<genome>.fa
data/<species>/bigwigs{_dataset}/<pseudobulk>.bw
data/<species>/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/<species>/negatives{_dataset}/<pseudobulk>_negatives.bed
data/<species>/peaks{_dataset}/differential_accessibility/da_peaks_tau.bed
data/<species>/peaks{_dataset}/differential_accessibility/da_peaks_pairwise.bed
models/<species>/{dataset/}<pseudobulk>/<pseudobulk>_f<fold>.torch
```

## Single Runs

Full-model fine-tuning:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    --dataset coarse \
    -p mesophyll \
    -f 0 \
    --mode full
```

Adapter fine-tuning:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    --dataset coarse \
    -p mesophyll \
    -f 0 \
    --mode adapter
```

The standard DA fractions are `0.5` and `0.8`:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    --dataset fine \
    -p bundle_sheath \
    -f 0 \
    --mode adapter \
    --da-fraction 0.8
```

## Standard Launcher

No SLURM integration: `launch.py` collects and prints all standard
full-model and adapter run commands whose expected training log is missing;
pipe into `parallel`/`simple_gpu_scheduler` or run directly.

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch.py
uv run python src/cherimoya_pipeline/da_finetuning/launch.py --dataset coarse
uv run python src/cherimoya_pipeline/da_finetuning/launch.py --dataset fine
```

Limit to one run group, pseudobulk, or fold:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch.py \
    --dataset coarse \
    --run adapter \
    --pseudobulk mesophyll \
    --fold 0
```

Include commands whose expected log already exists:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch.py \
    --dataset fine \
    --rerun-finished
```

Check expected logs without printing commands:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch.py \
    --dataset coarse \
    --check-finished
```

## Adapter Sweeps

Run a configured adapter sweep:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch_adapter_sweep.py \
    --dataset coarse \
    --config configs/adapter_sweeps/smoke.json
```

Include commands whose expected log already exists:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch_adapter_sweep.py \
    --dataset coarse \
    --rerun-finished
```

## Outputs

Base dataset outputs are written below `models/<species>/<pseudobulk>/`.
Coarse and fine outputs are written below `models/<species>/coarse/<pseudobulk>/`
and `models/<species>/fine/<pseudobulk>/`.

```text
finetuned/<pseudobulk>_f<fold>.torch
finetuned_<fraction>/<pseudobulk>_f<fold>.torch
adapters/<run_id>/<pseudobulk>_f<fold>.adapter.pt
adapter_models/<run_id>/<pseudobulk>_f<fold>.torch
```
