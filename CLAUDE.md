# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for training and evaluating Cherimoya models of chromatin accessibility, primarily in *Sorghum bicolor* (sorghum), with configs support extended to *Arabidopsis thaliana* and *Oryza sativa* (rice). Models are trained per pseudobulk cell type per species using 5-fold cross-validation with chromosome-based splits.

Cherimoya trains a single model end-to-end directly on raw signal — there is no separate bias-correction stage. The prior ChromBPNet-based pipeline (a two-stage bias model + accessibility model, trained in a separate TensorFlow environment) is preserved as-is on the `chrombpnet` branch.

**Species** (`configs/<species>/`, see `src/species_config.py`): default `S.bicolor` (6 pseudobulk cell types in `configs/pseudobulks.txt`: bundle_sheath, epidermis, guard, mesophyll, phloem, xylem) keeps today's flat, un-namespaced `configs/`/`data/` layout for backward compatibility; `A.thaliana` and `O.sativa` are namespaced under `configs/<species>/`, `data/<species>/`, etc. Species are identified by Latin binomial everywhere (`--species` flag / `$SPECIES` env var).

`A.thaliana` uses the `MoralesArabidopsis` root-atlas pseudobulk dataset (30 cell types in `configs/A.thaliana/pseudobulks.txt`, from `/oak/stanford/projects/sorghum/MoralesArabidopsis/pseudobulk_tracks/{peaks,fragments}/`) against the classic TAIR10 assembly (`data/A.thaliana/Athaliana_TAIR10.fa`, chromosomes `chr1`-`chr5`/`chrM`/`chrC` — see `src/data_preprocessing/download_arabidopsis_genome.sh`).

## Environment Setup

A single `plant-design` environment (uv-managed) now covers the whole pipeline:

```bash
mamba env create -f environment.yml
mamba activate plant-design
uv sync --frozen
```

`cherimoya` is a **Linux-only** core dependency — it requires `torch>=2.9`/`triton>=3.5.1`, and `triton` has no macOS wheels at all. Training (`fit/`) only runs on Linux GPU machines; preprocessing, benchmarking, attribution, and DA finetuning work in the same env on any platform once a model exists.

On HPC systems whose system toolchain can't build/resolve that requirement (e.g. Sherlock, see `pyproject.toml`'s `pybigtools==0.2.5` pin), the `fit/` and `benchmark/` SLURM launchers accept `--apptainer` to run jobs against a prebuilt Cherimoya Apptainer image instead of `uv run --frozen` (see `src/cherimoya_pipeline/apptainer/README.md`).

## Pipeline Execution Order

```bash
# 1. Preprocess: convert ATAC fragments → bigwigs
bash src/data_preprocessing/make_bigwigs.sh

# 2. Generate chromosome fold configs (if not already present)
python configs/write_folds.py [--species S.bicolor]

# 3. Generate GC-matched negative regions (Cherimoya's own `negatives` CLI; one per pseudobulk, not per fold)
uv run python src/cherimoya_pipeline/fit/negatives.py --interactive | parallel

# 4. Train Cherimoya models (all pseudobulks, all folds; single stage, no bias model)
uv run python src/cherimoya_pipeline/fit/launch.py

# 5. Benchmark models
uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p <pseudobulk> [-o]

# 6. Compute differential accessibility
python src/data_preprocessing/differential_accessibility.py

# 7. Benchmark DA predictions
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py

# 8. Finetune on DA peaks (one process per fold, e.g. one per GPU)
for fold in 0 1 2 3 4; do
    uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py -p <pseudobulk> -f $fold
done
```

All `fit/`/`benchmark/`/`attribute/`/`modisco/`/`da_finetuning/` launchers accept `--species` (default `S.bicolor`), `--dataset {base,fine,coarse}`, `--interactive` (print commands instead of submitting SLURM jobs), `--dry-run` (print sbatch scripts), and `--rerun`.

## Code Architecture

### `src/species_config.py`

Resolves `--species`/`$SPECIES` (default `S.bicolor`) to config/data/model/output directories. The default species keeps flat paths (`configs/`, `data/`, `models/`, ...); every other species is namespaced under a same-named subdirectory (`configs/A.thaliana/`, `data/A.thaliana/`, ...). Imported by essentially every script under `src/cherimoya_pipeline/` and `src/data_preprocessing/`.

### `src/cherimoya_pipeline/apptainer_launcher.py`

Shared `--apptainer`/`--apptainer-image`/`--apptainer-bind` CLI handling and run-prefix resolution (`uv run --frozen` by default, or `apptainer exec --nv` against the Cherimoya image) used by the `fit/` and `benchmark/` SLURM launchers.

### `src/cherimoya_pipeline/apptainer/`

`cherimoya.def`: Apptainer image definition bootstrapping from `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` and installing Cherimoya + daicon `--no-deps` on top, for HPC systems whose system toolchain can't build/resolve Cherimoya's `torch>=2.9`/`triton>=3.5.1` requirement directly.

### `src/cherimoya_pipeline/fit/`

- **`negatives.py`**: Submits SLURM jobs running Cherimoya's own `negatives` CLI (GC-matched sampling) to produce `data/<species>/negatives{_dataset}/<pseudobulk>_negatives.bed`, one per pseudobulk.
- **`train_cherimoya.py`**: The actual training entrypoint. Resolves our species/fold paths and hyperparameters into a `cherimoya_cli.defaults.default_fit_parameters`-shaped JSON file (saved alongside the model as `<pseudobulk>_f{fold}.fit_params.json`) and shells out to `cherimoya fit -p <file>`, rather than reimplementing its Muon/AdamW/SGD optimizer split and warmup+cosine schedules in Python. Reads `training_chroms`/`validation_chroms` from the existing `configs/<species>/fold_N.json` (`train`/`valid` keys); `test` is never touched here — held out entirely for `benchmark.py`. Saves to `models/<species>/<pseudobulk>/<pseudobulk>_f{fold}.torch` (Cherimoya's own config+state_dict checkpoint format, loaded via `Cherimoya.load()`); `cherimoya fit` also auto-runs its own evaluate step on `validation_chroms`, writing `<pseudobulk>_f{fold}.evaluate.json`/`.performance.tsv` as a side effect.
- **`launch.py`**: SLURM launcher submitting `train_cherimoya.py` runs per pseudobulk/fold.
- **`slurm_launcher.py`**: Shared job-submission helpers (species/dataset/pseudobulk/fold resolution, sbatch script generation, `--interactive`/`--dry-run`/`--apptainer` modes) used by both launchers above.

### `src/cherimoya_pipeline/benchmark/`

- **`benchmark.py`**: Loads trained Cherimoya models, runs genome-wide predictions on held-out test chromosomes, and computes profile Pearson/JSD and counts Pearson/Spearman per fold and aggregated. Outputs to `performance_metrics/cherimoya/<pseudobulk>.json`. Use `-o`/`--save-output` to save raw predictions to `predictions/cherimoya/<pseudobulk>.npz`.
- **`benchmark_da.py`**: Evaluates how well predicted counts reproduce cell-type specificity. Builds an out-of-fold predicted count matrix over the union peak set, converts to CPM, and computes log2FC / tau / pairwise log2FC correlations and specificity accuracy against observed values from `da_peaks.csv`. Outputs to `performance_metrics/differential_accessibility/da_benchmark.json`. Requires `differential_accessibility.py` to have been run first. Use `--save-output` for a per-peak CSV.
- **`model_variants.py`**: Shared `--finetuned`/`--adapter`/`--adapter-sweep-id` CLI handling and model-subdirectory resolution. `load_cherimoya_with_adapter()` loads a checkpoint via `Cherimoya.load()` and optionally injects a daicon adapter bundle at the model root (Cherimoya has no bias/accessibility submodel split to target).

### `configs/`

- **`species.json`** (default species) / **`<species>/species.json`**: manifest of `genome_fasta`/`chrom_sizes` filenames relative to `data_dir(species)`.
- **`chrom_splits.json`** (default species) / **`<species>/chrom_splits.json`**: chromosome-to-fold mapping.
- **`fold_N.json`**: per-fold train/valid/test chromosome assignments, generated by `write_folds.py`.
- **`write_folds.py`**: Generates `fold_N.json` files from `chrom_splits.json` using rolling 5-fold CV logic; takes `--species`.
- **`chrom_splits.yaml`**: reference chromosome-fold assignments for many species, keyed by Latin binomial, sourced from another project.
- **`import_chrom_splits.py`**: Converts one species' block from `chrom_splits.yaml` into `configs/<species>/chrom_splits.json`.

### `src/cherimoya_pipeline/attribute/`

DeepLIFT/SHAP attribution pipeline (model-agnostic `bpnetlite`/`tangermeme` machinery, unchanged from the ChromBPNet-based pipeline except for checkpoint loading):

- **`filter_nonACGT_regions.py`**: Filters peak BED files to remove regions whose centered 2114 bp window contains non-ACGT characters, is shorter than the window (near contig edges), or lies on a missing chromosome. Produces `data/<species>/peaks/{pseudobulk}_peaks_ACGT.narrowPeak`.
- **`save_ohe.py`**: Extracts and saves one-hot-encoded sequences for all peaks genome-wide to `attributions/<species>/{pseudobulk}_ohe.npz`. Run once per pseudobulk before attribution.
- **`da_peaks.py`**: Shared path-resolution helpers (species/dataset/pseudobulk → BED/model-dir/OHE paths) used by `save_ohe.py`, `deepliftshap.py`, and `modisco/`.
- **`deepliftshap.py`**: Computes hypothetical DeepLIFT/SHAP attributions across all peaks for a given pseudobulk and head (`profile`/`count`), loading the Cherimoya checkpoint via `model_variants.load_cherimoya_with_adapter`. Averages attributions across all 5 folds and saves to `attributions/<species>/cherimoya{_variant}/{pseudobulk}_{head}.npz`.
- **`launch.py`**: SLURM submission wrapper looping over pseudobulk × head.

### `src/cherimoya_pipeline/modisco/`

- **`launch.py`**: Shells out to standalone `modisco motifs` (tfmodisco-lite) on precomputed OHE/attribution npz files.
- **`launch_report.py`**: Shells out to `modisco report` to generate HTML reports from the `.modisco.h5` outputs.

### `src/cherimoya_pipeline/da_finetuning/`

- **`finetune_da.py`**: Finetunes the whole Cherimoya model (`--mode full`, default) or daicon adapters (`--mode adapter`) for one pseudobulk × fold on a mixed training set: 50% DA peaks (union of `da_peaks_tau.bed` + `da_peaks_pairwise.bed`) and 50% base mix (cell-type peaks at 7/8, nonpeaks at 1/8). Each DA peak is seen exactly once per epoch. Adapter targets use Cherimoya's own layer names (`iconv`/`blocks`/`fconv`/`linear`/`heads`, `daicon.Cherimoya_TARGETS`/`Cherimoya_PRESETS`); default target is `all_postblock` (Cherimoya has no single "all" preset). Requires `-p`/`--pseudobulk` and `-f`/`--fold`. Saves finetuned models to `models/<species>/<pseudobulk>/finetuned/`.
- **`finetune_adapters.py`**: Thin compatibility wrapper preserving the historical adapter-only CLI, delegating to `finetune_da.py --mode adapter`.
- **`launch.py`** / **`launch_adapter_sweep.py`**: SLURM submission wrappers (`simple_gpu_scheduler`-style, one process per GPU).

### `src/data_preprocessing/`

- **`make_bigwigs.sh`**: Converts per-pseudobulk fragment BAM files → strand-shifted bigwigs using `bam2bw`. Species-aware via `$SPECIES` (default `S.bicolor` keeps today's flat layout and hardcoded Stanford Oak cluster path defaults).
- **`differential_accessibility.py`**: Builds a union peak set from all pseudobulk narrowPeak files, extracts read counts from bigwigs, normalizes (CPM or TMM with `--tmm`), and computes per-peak specificity scores (log2FC, pairwise log2FC, tau). Outputs `da_peaks.csv`, `da_peaks_tau.bed`, `da_peaks_pairwise.bed`, per-cell-type ranked CSVs, and `summary.json` to `data/<species>/peaks/differential_accessibility/`. Must be run before `benchmark_da.py` or `finetune_da.py`. Model-agnostic; unchanged by the Cherimoya migration.

## Data Conventions

- Peaks: 10-column narrowPeak BED format. Negatives: 3-column BED, one file per pseudobulk (not per fold — Cherimoya's `negatives` CLI has no chromosome-split concept; fold filtering happens at train/predict time via `chroms=`).
- Accessibility signal: bigwig files
- Reference genome: resolved per-species via `species_config.genome_fasta()`; default species (`S.bicolor`) uses *S. bicolor* v5 (`data/Sbicolor_730_v5.0.fa`)
- Input window: 2114 bp; output window: 1000 bp
- Default random seed: 1234
- GPU scheduling: `simple_gpu_scheduler` used in `da_finetuning/` shell/launch scripts for parallel multi-GPU training
