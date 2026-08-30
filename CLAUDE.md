# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for training and evaluating Cherimoya models of chromatin accessibility, primarily in *Sorghum bicolor* (sorghum), with configs support extended to *Arabidopsis thaliana* and *Oryza sativa* (rice). Models are trained per pseudobulk cell type per species using 5-fold cross-validation with chromosome-based splits.

Cherimoya trains a single model end-to-end directly on raw signal — there is no separate bias-correction stage. The prior ChromBPNet-based pipeline (a two-stage bias model + accessibility model, trained in a separate TensorFlow environment) is preserved as-is on the `chrombpnet` branch.

**Species** (`configs/<species>/`, see `src/species_config.py`): default `S.bicolor` (6 pseudobulk cell types in `configs/S.bicolor/pseudobulks.txt`: bundle_sheath, epidermis, guard, mesophyll, phloem, xylem); every species, including the default, is namespaced under a same-named subdirectory (`configs/S.bicolor/`, `configs/A.thaliana/`, `data/S.bicolor/`, `data/A.thaliana/`, etc.) -- there is no flat/un-namespaced layout. Species are identified by Latin binomial everywhere (`--species` flag / `$SPECIES` env var).

`A.thaliana` uses the `MoralesArabidopsis` root-atlas pseudobulk dataset (30 cell types in `configs/A.thaliana/pseudobulks.txt`, from `/oak/stanford/groups/akundaje/ayhe/plant-design/data/A.thaliana/{peaks,fragments}/`) against the classic TAIR10 assembly (`data/A.thaliana/Athaliana_TAIR10.fa`, chromosomes `chr1`-`chr5`/`chrM`/`chrC` — see `src/data_preprocessing/download_arabidopsis_genome.sh`).

## Environment Setup

A single `plant-design` environment (uv-managed) now covers the whole pipeline:

```bash
mamba env create -f environment.yml
mamba activate plant-design
uv sync --frozen
```

`cherimoya` is a **Linux-only** core dependency — it requires `torch>=2.9`/`triton>=3.5.1`, and `triton` has no macOS wheels at all. Training (`fit/`) only runs on Linux GPU machines; preprocessing, benchmarking, attribution, and DA finetuning work in the same env on any platform once a model exists.

On Sherlock, every `cherimoya_pipeline` launcher (`fit/`, `benchmark/`, `attribute/`, `modisco/`, `da_finetuning/`) submits SLURM jobs that run via `apptainer exec` against the Cherimoya Apptainer image (`src/cherimoya_pipeline/apptainer/`) instead of `uv sync`ing the `plant-design` env directly on the login/compute node -- Sherlock's system toolchain can't reliably build/resolve `torch>=2.9`/`triton>=3.5.1`. See `src/cherimoya_pipeline/apptainer/README.md` for building the image and `check_gpu.py` for verifying it against Sherlock's GPU fleet.

## Pipeline Execution Order

```bash
# 1. Preprocess: convert ATAC fragments → bigwigs
bash src/data_preprocessing/make_bigwigs.sh | parallel

# 2. Generate chromosome fold configs (if not already present)
python configs/write_folds.py [--species S.bicolor]

# 3. Generate GC-matched negative regions (Cherimoya's own `negatives` CLI; one per pseudobulk, not per fold)
python src/cherimoya_pipeline/fit/negatives.py

# 4. Train Cherimoya models (all pseudobulks, all folds; single stage, no bias model)
python src/cherimoya_pipeline/fit/launch.py

# 5. Benchmark models
uv run python src/cherimoya_pipeline/benchmark/benchmark.py -p <pseudobulk> [-o]

# 6. Compute differential accessibility
uv run python src/data_preprocessing/differential_accessibility.py

# 7. Benchmark DA predictions
uv run python src/cherimoya_pipeline/benchmark/benchmark_da.py

# 8. Finetune on DA peaks (one process per fold, e.g. one per GPU)
for fold in 0 1 2 3 4; do
    uv run python src/cherimoya_pipeline/da_finetuning/finetune_da.py -p <pseudobulk> -f $fold
done
```

`fit/`/`benchmark/`/`attribute/`/`modisco/`/`da_finetuning/` launchers submit SLURM jobs for unfinished work (missing output files), running via `apptainer exec` against the Cherimoya Apptainer image (`--nv` for GPU jobs; see `src/cherimoya_pipeline/apptainer/`) rather than `uv run` directly. `--dry-run` prints the generated sbatch script instead of submitting; `--local` skips SLURM and runs in the foreground instead (combine with `--dry-run` to print one command per job, e.g. for `simple_gpu_scheduler`/`parallel` on a non-Sherlock GPU box). All accept `--species` (default `S.bicolor`), `--dataset {base,fine,coarse}`, and `--rerun` (submit/run jobs even if already done).

## Code Architecture

### `src/species_config.py`

Resolves `--species`/`$SPECIES` (default `S.bicolor`) to config/data/model/output directories. Every species is namespaced under a same-named subdirectory (`configs/S.bicolor/`, `data/S.bicolor/`, `configs/A.thaliana/`, `data/A.thaliana/`, ...); there is no flat/un-namespaced layout. Imported by essentially every script under `src/cherimoya_pipeline/` and `src/data_preprocessing/`.

### `src/cherimoya_pipeline/fit/`

- **`negatives.py`**: Collects and prints Cherimoya's own `negatives` CLI (GC-matched sampling) commands for pseudobulks missing a negatives BED, one per pseudobulk (`data/<species>/negatives{_dataset}/<pseudobulk>_negatives.bed`).
- **`train_cherimoya.py`**: The actual training entrypoint. Resolves our species/fold paths and hyperparameters into a `cherimoya_cli.defaults.default_fit_parameters`-shaped JSON file (saved alongside the model as `<pseudobulk>_f{fold}.fit_params.json`) and shells out to `cherimoya fit -p <file>`, rather than reimplementing its Muon/AdamW/SGD optimizer split and warmup+cosine schedules in Python. Reads `training_chroms`/`validation_chroms` from the existing `configs/<species>/fold_N.json` (`train`/`valid` keys); `test` is never touched here — held out entirely for `benchmark.py`. Saves to `models/<species>/<pseudobulk>/<pseudobulk>_f{fold}.torch` (Cherimoya's own config+state_dict checkpoint format, loaded via `Cherimoya.load()`); `cherimoya fit` also auto-runs its own evaluate step on `validation_chroms`, writing `<pseudobulk>_f{fold}.evaluate.json`/`.performance.tsv` as a side effect.
- **`launch.py`**: Collects and prints `train_cherimoya.py` commands per pseudobulk/fold missing a model checkpoint.
- **`launcher_common.py`**: Shared job-collection helpers (species/dataset/pseudobulk/fold resolution, `--rerun` handling, command printing) used by both launchers above.

### `src/cherimoya_pipeline/benchmark/`

- **`benchmark.py`**: Loads trained Cherimoya models, runs genome-wide predictions on held-out test chromosomes, and computes profile Pearson/JSD and counts Pearson/Spearman per fold and aggregated. Outputs to `performance_metrics/<species>/cherimoya/<pseudobulk>.json`. Use `-o`/`--save-output` to save raw predictions to `predictions/<species>/cherimoya/<pseudobulk>.npz`.
- **`benchmark_da.py`**: Evaluates how well predicted counts reproduce cell-type specificity. Builds an out-of-fold predicted count matrix over the union peak set, converts to CPM, and computes log2FC / tau / pairwise log2FC correlations and specificity accuracy against observed values from `da_peaks.csv`. Outputs to `performance_metrics/differential_accessibility/da_benchmark.json`. Requires `differential_accessibility.py` to have been run first. Use `--save-output` for a per-peak CSV.
- **`model_variants.py`**: Shared `--finetuned`/`--adapter`/`--adapter-sweep-id` CLI handling and model-subdirectory resolution. `load_cherimoya_with_adapter()` loads a checkpoint via `Cherimoya.load()` and optionally injects a daicon adapter bundle at the model root (Cherimoya has no bias/accessibility submodel split to target).
- **`launch.py`**: Submits one SLURM job per model variant, bundling its cross-pseudobulk `benchmark.py` calls plus `benchmark_da.py` and (base dataset only) `pairwise_correlations_predicted.py` into a single bash script that re-checks each step's own output-file completion marker at runtime (not just at submission time) -- mirrors `fit/launch.py`'s per-pseudobulk fold loop, so a `--requeue`'d rerun after pre-emption on the preemptible partitions skips whatever steps already finished and only redoes the rest. `--print-commands` prints every not-yet-finished step's command to stdout, prefixed with `uv run` instead of an apptainer wrapper (no SLURM, no bundling) for hand-picking on a cluster with the `cherimoya` extra installed or inside an already-entered Apptainer shell; combine with `--rerun` to print every step regardless. Job names/log filenames include the species for easy `scancel`/`squeue` filtering.

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
- **`launch.py`**: Submits one SLURM job per (pseudobulk, head) via `deepliftshap.py`, skipping ones whose output already exists or whose inputs are missing.

### `src/cherimoya_pipeline/modisco/`

- **`launch.py`**: Submits one SLURM job per (pseudobulk, head) running `modisco motifs` (tfmodisco-lite) on precomputed OHE/attribution npz files, via the Cherimoya Apptainer image (CPU-only; `modisco` is pip-installed inside that image, not the `plant-design` uv env).
- **`launch_report.py`**: Submits one SLURM job per (pseudobulk, head) running `modisco report` to generate HTML reports from the `.modisco.h5` outputs, same Apptainer-image path as `launch.py`.

### `src/cherimoya_pipeline/da_finetuning/`

- **`finetune_da.py`**: Finetunes the whole Cherimoya model (`--mode full`, default) or daicon adapters (`--mode adapter`) for one pseudobulk × fold on a mixed training set: 50% DA peaks (union of `da_peaks_tau.bed` + `da_peaks_pairwise.bed`) and 50% base mix (cell-type peaks at 7/8, nonpeaks at 1/8). Each DA peak is seen exactly once per epoch. Adapter targets use Cherimoya's own layer names (`iconv`/`blocks`/`fconv`/`linear`/`heads`, `daicon.Cherimoya_TARGETS`/`Cherimoya_PRESETS`); default target is `all_postblock` (Cherimoya has no single "all" preset). Requires `-p`/`--pseudobulk` and `-f`/`--fold`. Saves finetuned models to `models/<species>/<pseudobulk>/finetuned/`.
- **`finetune_adapters.py`**: Thin compatibility wrapper preserving the historical adapter-only CLI, delegating to `finetune_da.py --mode adapter`.
- **`launch.py`**: Submits one SLURM job per pseudobulk, bundling every selected run group x fold x DA-fraction step into a single bash script that re-checks each step's own training-log completion marker at runtime (not just at submission time) -- mirrors `fit/launch.py`'s per-pseudobulk fold loop, so a `--requeue`'d rerun after pre-emption on the preemptible partitions skips whatever steps already finished and only redoes the rest. `--print-commands` prints every not-yet-finished step's command to stdout, prefixed with `uv run` instead of an apptainer wrapper (no SLURM, no bundling) for hand-picking on a cluster with the `cherimoya` extra installed (`uv sync --extra cherimoya`) or inside an already-entered Apptainer shell; combine with `--rerun-finished` to print every step regardless. Job names/log filenames include the species and, when `--run` narrows the default, the selected run group(s), for easy `scancel`/`squeue` filtering by species.
- **`launch_adapter_sweep.py`**: Submits one SLURM job per (pseudobulk, fold, hyperparameter combo) for the full adapter hyperparameter sweep -- kept per-combo (not consolidated like `launch.py`) since sweep combos are independent and benefit from scheduling in parallel rather than serially in one job. `--local --dry-run` on either launcher prints plain commands for multi-GPU fan-out via `simple_gpu_scheduler` instead.

### `src/cherimoya_pipeline/apptainer/`

- **`cherimoya.def`**: Apptainer definition bootstrapping from `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` and pip-installing Cherimoya plus every non-torch dependency (including `modisco`/tf-modisco-lite) `--no-deps` on top of the base image's own torch build. Built into a `.sif` and run via `apptainer exec --nv`, this is the default execution path for every launcher above -- sidesteps Sherlock's old system toolchain, which can't reliably build/resolve `torch>=2.9`/`triton>=3.5.1` directly.
- **`check_gpu.py`**: Submits one short per-GPU-SKU SLURM job that loads the image and moves a tensor onto the GPU, since Sherlock has no way to query driver/CUDA compatibility ahead of time.

**The image actually used for training is defined outside this repo**, at
[`adamyhe/sherlock`](https://github.com/adamyhe/sherlock) (`cherimoya/cherimoya.def`); the in-repo
`cherimoya.def` above is a separate, divergent definition. Launchers resolve the image purely by
path (`$CHERIMOYA_APPTAINER_IMAGE`, default `/scratch/users/$USER/apptainer/cherimoya.sif`), so
they use whichever image is built there regardless of which definition produced it. Two
consequences worth knowing:

- **`daicon` is deliberately not installed in the production image.** `attribute/deepliftshap.py`
  and `da_finetuning/finetune_da.py` import it at module top level, so both fail immediately
  (`ModuleNotFoundError`) under `apptainer exec` against that image -- run them in the
  `plant-design` uv env (`uv sync --extra cherimoya`, where `daicon` is a base dependency)
  instead. `benchmark/benchmark.py` is unaffected: its `daicon` import is lazy, inside
  `model_variants.load_cherimoya_with_adapter()`, so only `--adapter` runs need it. This is the
  intended split -- models are trained in the Apptainer image on Sherlock, and DA finetuning /
  attribution run in the uv env on the lab cluster.
- **The production image installs Cherimoya from a git commit, not PyPI.** That commit is
  upstream's v0.2.1, which was never released to PyPI (PyPI stops at 0.2.0), so the `cherimoya`
  uv extra pins the same commit to keep the two environments on identical code -- see the comment
  on that extra in `pyproject.toml`.


### `src/data_preprocessing/`

- **`make_bigwigs.sh`**: Resolves paths and prints one `bam2bw -f` command per pseudobulk (both Tn5 insertion sites per fragment) to stdout -- pipe into `parallel`/`simple_gpu_scheduler` to run. This repo's checkout lives directly on Oak, so by default fragments/genome are read straight out of `data/<species>/fragments{_fine,_coarse}/` and `data/<species>/` -- no separate absolute Oak path needed (override via `DATA_DIR`/`REF_DIR` for sources living elsewhere). Species-aware via `$SPECIES` (default `S.bicolor`) and dataset-aware via `$DATASET` (`base`/`fine`/`coarse`, default `base`). Outputs to `data/<species>/bigwigs{_fine,_coarse}/`.
- **`differential_accessibility.py`**: Builds a union peak set from all pseudobulk narrowPeak files, extracts read counts from bigwigs, normalizes (CPM or TMM with `--tmm`), and computes per-peak specificity scores (log2FC, pairwise log2FC, tau). Outputs `da_peaks.csv`, `da_peaks_tau.bed`, `da_peaks_pairwise.bed`, per-cell-type ranked CSVs, and `summary.json` to `data/<species>/peaks/differential_accessibility/`. Must be run before `benchmark_da.py` or `finetune_da.py`. Model-agnostic; unchanged by the Cherimoya migration.

## Data Conventions

- Peaks: 10-column narrowPeak BED format. Negatives: 3-column BED, one file per pseudobulk (not per fold — Cherimoya's `negatives` CLI has no chromosome-split concept; fold filtering happens at train/predict time via `chroms=`).
- Accessibility signal: bigwig files
- Reference genome: resolved per-species via `species_config.genome_fasta()`; default species (`S.bicolor`) uses *S. bicolor* v5 (`data/S.bicolor/Sbicolor_730_v5.0.fa`)
- Input window: 2114 bp; output window: 1000 bp
- Default random seed: 1234
- GPU scheduling: SLURM (via each launcher's `apptainer exec --nv` sbatch submission) is the default; `--local --dry-run` on any launcher prints plain commands for `simple_gpu_scheduler`-based multi-GPU fan-out instead
