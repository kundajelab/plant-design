# Cherimoya Benchmarking

Run commands from the repository root. Benchmarking uses the `plant-design`
Python environment through `uv`.

## Inputs

```text
data/<species>/<genome>.fa
data/<species>/bigwigs{_dataset}/<pseudobulk>.bw
data/<species>/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/<species>/peaks{_dataset}/differential_accessibility/da_peaks.csv
models/<species>/{dataset/}<pseudobulk>/<pseudobulk>_f<fold>.torch
configs/<species>/pseudobulks{_dataset}.txt
configs/<species>/fold_0.json ... fold_4.json
```

`--species` defaults to `S.bicolor`. For non-base datasets, outputs add the
dataset suffix, for example `performance_metrics/<species>/cherimoya_coarse/`.

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
performance_metrics/<species>/differential_accessibility/da_benchmark.json
performance_metrics/<species>/differential_accessibility_coarse/da_benchmark.json
performance_metrics/<species>/differential_accessibility_fine/da_benchmark.json
```

## Launcher

`launch.py` submits one SLURM job per variant, bundling that variant's
cross-pseudobulk/DA/pairwise steps into a single sbatch script that runs via
`apptainer exec --nv` against the Cherimoya Apptainer image (see
`../apptainer/`). `--dry-run` prints the generated sbatch script instead of
submitting; `--local` runs in the foreground instead of submitting to
SLURM (combine with `--dry-run` to print one command per step for
`simple_gpu_scheduler`/`parallel`):

```bash
python src/cherimoya_pipeline/benchmark/launch.py
python src/cherimoya_pipeline/benchmark/launch.py --dry-run
```

For coarse or fine datasets, the launcher defaults to the base model variant:

```bash
python src/cherimoya_pipeline/benchmark/launch.py --dataset coarse
python src/cherimoya_pipeline/benchmark/launch.py --dataset fine
```

Print jobs even if their output already exists:

```bash
python src/cherimoya_pipeline/benchmark/launch.py \
    --dataset coarse \
    --rerun
```

Use `--variant` to select a trained variant on the base dataset:

```bash
python src/cherimoya_pipeline/benchmark/launch.py --variant finetuned
python src/cherimoya_pipeline/benchmark/launch.py --variant adapter_0.8
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
