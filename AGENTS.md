# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Research pipeline for training and evaluating ChromBPNet models of chromatin accessibility in *Sorghum bicolor* (sorghum). Models are trained per pseudobulk cell type using 5-fold cross-validation with chromosome-based splits.

**6 pseudobulk cell types** (configs/pseudobulks.txt): bundle_sheath, epidermis, guard, mesophyll, phloem, xylem

**Two-stage training**:
1. Bias model (models Tn5/DNASE insertion bias)
2. Accessibility (ChromBPNet) model with bias factored out

## Environment Setup

Two separate environments are required (TF 2.8 and modern PyTorch are incompatible):

**TensorFlow env** (`chrombpnet`) — ChromBPNet training (steps 3–5):
```bash
mamba create -n chrombpnet python=3.8
mamba activate chrombpnet
mamba install -y -c conda-forge -c bioconda samtools bedtools ucsc-bedgraphtobigwig pybigwig meme
mamba install -c conda-forge -c nvidia "cudnn>=8.9,<9" cudatoolkit
pip install tensorflow==2.8.0 chrombpnet
export NUMBA_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/chrombpnet/lib
```

**PyTorch env** (`plant-design`) — preprocessing, conversion, and benchmarking (steps 1–2, 6–7):
```bash
mamba env create -f environment.yml
mamba activate plant-design
uv sync --frozen
```

## Pipeline Execution Order

```bash
# 1. Preprocess: convert ATAC fragments → bigwigs
bash src/data_preprocessing/make_bigwigs.sh

# 2. Generate chromosome fold configs (if not already present)
python configs/write_folds.py

# 3. Generate non-peak background regions
python src/chrombpnet/fit/nonpeaks.py --interactive | parallel

# 4. Train bias models (all pseudobulks, all folds)
python src/chrombpnet/fit/bias.py

# 5. Train accessibility models
python src/chrombpnet/fit/accessibility.py

# 6. Convert TensorFlow → PyTorch
uv run python src/chrombpnet/fit/convert_to_torch.py

# 7. Benchmark models
uv run python src/chrombpnet/benchmark/benchmark.py -p <pseudobulk> [-o]

# 8. Compute differential accessibility
python src/data_preprocessing/differential_accessibility.py

# 9. Benchmark DA predictions
python src/chrombpnet/benchmark/benchmark_da.py

# 10. Finetune on DA peaks (one process per fold, e.g. one per GPU)
for fold in 0 1 2 3 4; do
    python src/chrombpnet/da_finetuning/finetune_da.py -p <pseudobulk> -f $fold
done
```

## Code Architecture

### `src/chrombpnet/fit/`

The main training module, adapted from [kundajelab/chrombpnet](https://github.com/kundajelab/chrombpnet):

- **`cli.py`**: Entry point. Dispatches subcommands: `pipeline/train/qc/bias/prep/pred_bw/contribs_bw/modisco_motifs/footprints/snp_score`. Creates the output directory structure (`logs/`, `auxiliary/`, `models/`, `evaluation/`).
- **`parsers.py`**: All argument definitions. Key inputs: genome FASTA, chrom sizes, bigwig signals, peak/nonpeak BED files (10-column format), fold JSON.
- **`pipelines.py`**: Core training logic. `chrombpnet_train_pipeline()` runs end-to-end training + QC + evaluation. `train_bias_pipeline()` handles bias-only training. Both call into the chrombpnet library for TF/Keras model training, SHAP contribution scores, modisco motifs, and marginal footprinting.
- **`convert_to_torch.py`**: Post-training conversion. Converts 3 models per fold (bias, no-bias accessibility, combined ChromBPNet) using bpnetlite. Processes 5 folds × N pseudobulks.

### `src/chrombpnet/benchmark/`

- **`benchmark.py`**: Loads converted PyTorch models, runs genome-wide predictions, and computes profile Pearson/JSD and counts Pearson/Spearman per fold and aggregated. Outputs to `performance_metrics/chrombpnet/<pseudobulk>.json`. Use `-o`/`--save-output` to save raw predictions to `predictions/chrombpnet/<pseudobulk>.npz`.
- **`benchmark_da.py`**: Evaluates how well predicted counts reproduce cell-type specificity. Builds an out-of-fold predicted count matrix over the union peak set, converts to CPM, and computes log2FC / tau / pairwise log2FC correlations and specificity accuracy against observed values from `da_peaks.csv`. Outputs to `performance_metrics/differential_accessibility/da_benchmark.json`. Requires `differential_accessibility.py` to have been run first. Use `--save-output` for a per-peak CSV.

### `configs/`

- **`chrom_splits.json`**: Chromosome-to-fold mapping (folds 0–4, covering Chr01–Chr10).
- **`fold_N.json`**: Per-fold train/valid/test chromosome assignments used by chrombpnet training.
- **`write_folds.py`**: Generates fold_N.json files from chrom_splits.json using rolling 5-fold CV logic.

### `src/chrombpnet/attribute/`

DeepLIFT/SHAP attribution pipeline:

- **`filter_nonACGT_regions.py`**: Filters peak BED files to remove regions whose centered 2114 bp window contains non-ACGT characters, is shorter than the window (near contig edges), or lies on a missing chromosome. Produces `data/peaks/{pseudobulk}_peaks_ACGT.narrowPeak`.
- **`save_ohe.py`**: Extracts and saves one-hot-encoded sequences for all peaks genome-wide to `attributions/{pseudobulk}_ohe.npz`. Run once per pseudobulk before attribution.
- **`deepliftshap.py`**: Computes hypothetical DeepLIFT/SHAP attributions across all peaks for a given pseudobulk, model type (`bias`/`accessibility`), and head (`profile`/`count`). Averages attributions across all 5 folds and saves to `attributions/{pseudobulk}_{model_type}_{head}.npz`.
- **`launch.sh`**: SLURM array job (`--array=0-7`) that runs `deepliftshap.py` for 4 pseudobulks × 2 heads in parallel, one GPU per task. Requires `$REPO_ROOT` to be exported before calling `sbatch`.
- **`cmd.sh`**: Local multi-GPU alternative using `simple_gpu_scheduler`. Also contains the `parallel`-based commands for running `filter_nonACGT_regions.py` and `save_ohe.py`.

### `src/chrombpnet/da_finetuning/`

- **`finetune_da.py`**: Finetunes the accessibility submodel (bias frozen) for one pseudobulk × fold on a mixed training set: 50% DA peaks (union of `da_peaks_tau.bed` + `da_peaks_pairwise.bed`) and 50% base mix (cell-type peaks at 7/8, nonpeaks at 1/8). Each DA peak is seen exactly once per epoch. Requires `-p`/`--pseudobulk` and `-f`/`--fold`. Saves finetuned models to `models/{pseudobulk}/finetuned/`.

### `src/data_preprocessing/`

- **`make_bigwigs.sh`**: Converts per-pseudobulk fragment BAM files → strand-shifted bigwigs using `bam2bw`. References Stanford Oak cluster paths.
- **`differential_accessibility.py`**: Builds a union peak set from all pseudobulk narrowPeak files, extracts read counts from bigwigs, normalizes (CPM or TMM with `--tmm`), and computes per-peak specificity scores (log2FC, pairwise log2FC, tau). Outputs `da_peaks.csv`, `da_peaks_tau.bed`, `da_peaks_pairwise.bed`, per-cell-type ranked CSVs, and `summary.json` to `data/peaks/differential_accessibility/`. Must be run before `benchmark_da.py` or `finetune_da.py`.

## Data Conventions

- Peaks and non-peaks: 10-column BED format
- Accessibility signal: bigwig files
- Reference genome: *S. bicolor* v5 (`Sbicolor_730_v5.0.fa`)
- Input window: 2114 bp; output window: 1000 bp
- Default random seed: 1234
- GPU scheduling: `simple_gpu_scheduler` used in shell scripts for parallel multi-GPU training
