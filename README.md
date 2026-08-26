# plant-design

Regulatory sequence modeling and enhancer/promoter design in plants. Trains and evaluates [Cherimoya](https://github.com/jmschrei/cherimoya) models of chromatin accessibility, primarily in *Sorghum bicolor*, using 5-fold cross-validation with chromosome-based splits across 6 pseudobulk cell types (bundle_sheath, epidermis, guard, mesophyll, phloem, xylem). Config support is also extended to *Arabidopsis thaliana* and *Oryza sativa* (rice) — see `src/species_config.py`.

Cherimoya trains a single model end-to-end directly on raw signal, with no separate bias-correction stage. The prior [ChromBPNet](https://github.com/kundajelab/chrombpnet)-based pipeline (a two-stage bias model + accessibility model, trained in a separate TensorFlow environment) is preserved as-is on the `chrombpnet` branch.

## Installation

A single environment now covers the whole pipeline:

```bash
mamba env create -f environment.yml
mamba activate plant-design
uv sync --frozen
```

Python dependencies for preprocessing, training, benchmarking, attribution,
sequence design, and adapter fine-tuning are managed in `pyproject.toml` and
locked in `uv.lock`. Run entrypoints through `uv run`, for example
`uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p mesophyll`.

`cherimoya` is a **Linux-only** dependency: it requires `torch>=2.9`/
`triton>=3.5.1`, and `triton` has no macOS wheels at all. Training
(`src/cherimoya_pipeline/fit/`) only runs on Linux GPU machines; everything
downstream of a trained model (benchmarking, attribution, DA finetuning) runs
in the same env on any platform.

## Pipeline

### 1. Data preprocessing

Convert ATAC fragment files to per-pseudobulk bigwigs:

```bash
bash src/data_preprocessing/make_bigwigs.sh | parallel
DATASET=fine bash src/data_preprocessing/make_bigwigs.sh | parallel
DATASET=coarse bash src/data_preprocessing/make_bigwigs.sh | parallel
```

Expected data layout under `data/<species>/` (default species: `S.bicolor`):
```
data/S.bicolor/
├── Sbicolor_730_v5.0.fa          # Sorghum v5 reference genome
├── peaks/<pseudobulk>_peaks.narrowPeak
├── peaks_fine/<pseudobulk>_peaks.narrowPeak
├── peaks_coarse/<pseudobulk>_peaks.narrowPeak
├── bigwigs/<pseudobulk>.bw
├── bigwigs_fine/<pseudobulk>.bw
└── bigwigs_coarse/<pseudobulk>.bw
```

Every species (including the default) is namespaced under `data/<species>/`
(see `src/species_config.py`); pass `--species <species>` to any Python
entrypoint, or set `$SPECIES` for shell scripts.

The base dataset is the default for training and analysis scripts. Use
`--dataset fine` or `--dataset coarse` with Python entrypoints such as the
job launchers, `benchmark.py`, and DeepLIFT/SHAP scripts.

### 2. Generate chromosome fold configs

```bash
python configs/write_folds.py
python configs/write_folds.py --species A.thaliana
```

Generates `configs/<species>/fold_0.json` ... `fold_4.json` from
`configs/<species>/chrom_splits.json` using rolling 5-fold CV logic.

### 3. Generate negative regions

Cherimoya samples GC-matched negatives itself via its own `negatives` CLI, one
BED file per pseudobulk (not per fold — chromosome-split filtering happens at
train/predict time, not at negative-sampling time):

```bash
uv run python src/cherimoya_pipeline/fit/negatives.py | parallel
uv run python src/cherimoya_pipeline/fit/negatives.py --dataset fine
```

### 4. Train Cherimoya models

Trains one model per pseudobulk per fold directly against raw bigwig signal —
no bias model, no separate accessibility stage:

```bash
uv run python src/cherimoya_pipeline/fit/launch.py --dataset fine
```

There's no SLURM integration: `launch.py` collects pseudobulk/fold jobs with
a missing model checkpoint and prints one local command per job to stdout,
so it can be piped to `parallel`, `simple_gpu_scheduler`, or another local
scheduler. `train_cherimoya.py` is the actual training entrypoint if you
want to run a single pseudobulk/fold directly for debugging.

Models are saved to `models/<species>/<pseudobulk>/<pseudobulk>_f<fold>.torch`
(Cherimoya's own config+state_dict checkpoint format, loaded via
`Cherimoya.load()`).

### 5. Benchmark

Evaluates models on held-out test chromosomes per fold:

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p <pseudobulk> [-o]
uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p <pseudobulk> --adapter --da-fraction 0.8
```

Options:
- `-p` / `--pseudobulk`: one of `bundle_sheath`, `epidermis`, `guard`, `mesophyll`, `phloem`, `xylem`
- `-o` / `--save-output`: save scaled predictions and signals to `predictions/<species>/cherimoya/<pseudobulk>.npz`
- `-b` / `--batch-size`: inference batch size (default: 64)
- `--adapter`: benchmark the default daicon adapter run for the selected `--da-fraction`
- `--adapter-sweep-id` / `--adapter-run-id`: benchmark an explicit adapter run ID

Metrics (profile Pearson, Jensen-Shannon distance, log-counts Pearson, counts Spearman) are saved to `performance_metrics/<species>/cherimoya/<pseudobulk>.json`.

### 6. Differential accessibility

Build the union peak set, extract observed accessibility counts, and compute DA
specificity metrics:

```bash
uv run python src/data_preprocessing/differential_accessibility.py
uv run python src/data_preprocessing/differential_accessibility.py --dataset coarse
uv run python src/data_preprocessing/differential_accessibility.py --dataset fine
```

Outputs are written under the matching peaks namespace:

```text
data/<species>/peaks/differential_accessibility/
data/<species>/peaks_coarse/differential_accessibility/
data/<species>/peaks_fine/differential_accessibility/
```

These outputs are required for DA benchmarking and adapter fine-tuning.

Evaluate whether out-of-fold model predictions recover observed cell-type
specificity:

```bash
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py [--save-output]
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py --dataset coarse [--save-output]
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py --dataset fine [--save-output]
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py --adapter --da-fraction 0.8
```

### 7. Full-model DA fine-tuning

The default DA fine-tuning path updates the whole Cherimoya model (there is no
bias submodel to keep frozen):

```bash
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py -p <pseudobulk> -f <fold> -v
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    --dataset coarse -p <pseudobulk> -f <fold> -v
```

Fine-tuned models are saved under `models/<species>/<pseudobulk>/finetuned/`.
For `--dataset coarse` or `--dataset fine`, outputs are saved under
`models/<species>/coarse/<pseudobulk>/...` or `models/<species>/fine/<pseudobulk>/...`.
Use `--da-fraction 0.8` to save the 80:20 DA replay run under
`models/<species>/<pseudobulk>/finetuned_0.8/`.

### 8. Low-rank adapter DA fine-tuning

Adapter fine-tuning freezes the base Cherimoya model and trains only small
low-rank residual adapters from `daicon`. Adapter bundles are saved separately
and are loaded onto the matching base fold model at prediction/attribution time.
The frozen base weights are not included in the adapter state dict; only
adapter branch weights are saved.

The main DA fine-tuning entrypoint supports daicon adapter mode. Adapter
targets use Cherimoya's own layer names (`iconv`, `blocks`, `fconv`, `linear`,
`heads` — see `daicon.Cherimoya_TARGETS`/`Cherimoya_PRESETS`); the default is
`all_postblock` (Cherimoya has no single "all" preset the way BPNet does),
with `conv_rank=16`, `linear_rank=4`, dropout `0`, and alpha-8 scaling.

```bash
uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    -p <pseudobulk> -f <fold> --mode adapter --da-fraction 0.5 -v

uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py \
    -p <pseudobulk> -f <fold> --mode adapter --da-fraction 0.8 -v
```

Artifacts:

```text
models/<species>/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.adapter.pt
models/<species>/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.config.json
models/<species>/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.log.tsv
models/<species>/<pseudobulk>/adapters/daicon_default_da0p8/<pseudobulk>_f<fold>.adapter.pt
```

The `.adapter.pt` bundle contains a manifest plus adapter-only `state_dict`.
The manifest records the pseudobulk, fold, base model path, adapter config,
layer specs, trainable parameter count, and training hyperparameters. Use the
final `.log.tsv` as the completion marker because the best adapter checkpoint is
written during training before the loop finishes.

The legacy `finetune_adapters.py` CLI remains available as a compatibility
wrapper around `finetune_da.py --mode adapter`.

Use the standard launcher to collect and print full-model DA fine-tuning plus
daicon adapter run commands, with both 0.5 and 0.8 DA fractions by default
(no SLURM integration -- pipe into `parallel`/`simple_gpu_scheduler` or run
directly):

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch.py \
    --run full \
    --run adapter

uv run python src/cherimoya_pipeline/da_finetuning/launch.py \
    --dataset coarse \
    --run full \
    --run adapter
```

Use the sweep launcher for the broad adapter hyperparameter grid. It prints
one command per pseudobulk/fold/hyperparameter combination and uses the
final `.log.tsv` as the completion marker:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json
```

Check finished adapter jobs:

```bash
uv run python src/cherimoya_pipeline/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json \
    --check-finished
```

Benchmark the active model set (base, both full-finetuned models, and both
default daicon adapter models):

```bash
uv run python src/cherimoya_pipeline/benchmark/launch.py
```

Summarize and plot the active model benchmark outputs:

```bash
uv run python src/cherimoya_pipeline/benchmark/plot_benchmark.py
```

> The LoRA sweep benchmark/plotting scripts (`benchmark_lora_sweep.py`,
> `plot_lora_sweep.py`, `select_best_lora_model.py`, `plot_celltype_scatter.py`,
> `pairwise_correlations.py`) are development-only utilities that still assume
> ChromBPNet's checkpoint format and directory labels; they haven't been
> migrated to Cherimoya yet.

### 9. Attribution

Attribution and tf-MoDISco have directory-local runbooks:

- `src/cherimoya_pipeline/attribute/README.md`: peak filtering, OHE extraction, and
  DeepLIFT/SHAP launch commands for base/fine/coarse and DA-specific runs.
- `src/cherimoya_pipeline/modisco/README.md`: motif discovery and report launch
  commands for base, finetuned, adapter, and DA-specific attribution outputs.
