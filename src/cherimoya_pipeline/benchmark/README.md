# Cherimoya Benchmarking

Run commands from the repository root. Benchmarking uses the `plant-design`
Python environment through `uv`.

## Inputs

```text
data/Sbicolor_730_v5.0.fa
data/bigwigs{_dataset}/<pseudobulk>.bw
data/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/peaks{_dataset}/differential_accessibility/da_peaks.csv
models/{dataset/}<pseudobulk>/<pseudobulk>_f<fold>.torch
configs/pseudobulks{_dataset}.txt
configs/fold_0.json ... configs/fold_4.json
```

For non-base datasets, outputs add the dataset suffix, for example
`performance_metrics/cherimoya_coarse/`.

## Single Model Benchmarks

Run one pseudobulk:

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p mesophyll
uv run python src/cherimoya_pipeline/benchmark/benchmark.py --dataset coarse -p mesophyll
uv run python src/cherimoya_pipeline/benchmark/benchmark.py --dataset fine -p bundle_sheath
```

Save raw predictions with `--save-output`:

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark.py \
    --dataset coarse \
    -p mesophyll \
    --save-output
```

## Differential Accessibility Benchmarks

Run `src/data_preprocessing/differential_accessibility.py` first for the same
dataset.

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py --dataset coarse
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py --dataset fine
```

Main outputs:

```text
performance_metrics/differential_accessibility/da_benchmark.json
performance_metrics/differential_accessibility_coarse/da_benchmark.json
performance_metrics/differential_accessibility_fine/da_benchmark.json
```

## Launcher

Submit the standard benchmark set:

```bash
uv run python src/cherimoya_pipeline/benchmark/launch.py
```

For coarse or fine datasets, the launcher defaults to the base model variant:

```bash
uv run python src/cherimoya_pipeline/benchmark/launch.py --dataset coarse
uv run python src/cherimoya_pipeline/benchmark/launch.py --dataset fine
```

Print the commands instead of submitting jobs:

```bash
uv run python src/cherimoya_pipeline/benchmark/launch.py \
    --dataset coarse \
    --interactive \
    --rerun
```

Use `--variant` to select a trained variant on the base dataset:

```bash
uv run python src/cherimoya_pipeline/benchmark/launch.py --variant finetuned
uv run python src/cherimoya_pipeline/benchmark/launch.py --variant adapter_0.8
```

## Plotting

After metrics are present, generate summary figures:

```bash
uv run python src/cherimoya_pipeline/benchmark/plot_benchmark.py
uv run python src/cherimoya_pipeline/benchmark/plot_celltype_scatter.py
```

Adapter sweep utilities live here too:

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark_lora_sweep.py --interactive
uv run python src/cherimoya_pipeline/benchmark/select_best_lora_model.py
uv run python src/cherimoya_pipeline/benchmark/plot_lora_sweep.py
```
