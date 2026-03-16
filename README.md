# plant-design

Regulatory sequence modeling and enhancer/promoter design in plants. Trains and evaluates [ChromBPNet](https://github.com/kundajelab/chrombpnet) models of chromatin accessibility in *Sorghum bicolor*, using 5-fold cross-validation with chromosome-based splits across 6 pseudobulk cell types (bundle_sheath, epidermis, guard, mesophyll, phloem, xylem).

## Installation

### Tensorflow environment

Create a conda environment and install all dependencies for ChromBPNet training:

```bash
mamba create -n chrombpnet python=3.8
mamba activate chrombpnet
mamba install -y -c conda-forge -c bioconda samtools bedtools ucsc-bedgraphtobigwig pybigwig meme
mamba install -c conda-forge -c nvidia "cudnn>=8.9,<9" cudatoolkit
pip install tensorflow==2.8.0 chrombpnet
```

Export the following variables before running ChromBPNet training:

```bash
export NUMBA_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/envs/chrombpnet/lib
```

Also note that TF installation is a bit of a pain. These things have worked for me, but you may need to mess around with dependencies and install orders to get things to actually work.

### PyTorch environment

All upstream and downstream processing from model training requires a separate PyTorch environment, as TensorFlow 2.8 and modern PyTorch are not compatible in the same environment:

```bash
mamba create -n torch python=3.13
mamba activate torch
pip install bpnetlite tangermeme tqdm bam2bw ledidi
```

## Pipeline

### 1. Data preprocessing

Convert ATAC fragment files to per-pseudobulk bigwigs:

```bash
bash src/data_preprocessing/make_bigwigs.sh
```

Expected data layout under `data/`:
```
data/
├── Sbicolor_730_v5.0.fa          # Sorghum v5 reference genome
├── peaks/<pseudobulk>_peaks.narrowPeak
└── bigwigs/<pseudobulk>.bw
```

### 2. Generate non-peak background regions

```bash
bash src/chrombpnet/fit/nonpeaks.sh
```

### 3. Train bias models

Trains Tn5 insertion bias models for all pseudobulks across all 5 folds:

```bash
bash src/chrombpnet/fit/bias.sh
```

### 4. Train accessibility (ChromBPNet) models

```bash
bash src/chrombpnet/fit/accessibility.sh
```

### 5. Convert to PyTorch

Converts trained TensorFlow `.h5` models to PyTorch `.torch` format:

```bash
python src/chrombpnet/fit/convert_to_torch.py
```

### 6. Benchmark

Evaluates models on held-out test chromosomes per fold:

```bash
python src/chrombpnet/benchmark/benchmark.py -p <pseudobulk> [-o]
```

Options:
- `-p` / `--pseudobulk`: one of `bundle_sheath`, `epidermis`, `guard`, `mesophyll`, `phloem`, `xylem`
- `-o` / `--save-output`: save scaled predictions and signals to `predictions/chrombpnet/<pseudobulk>.npz`
- `-b` / `--batch-size`: inference batch size (default: 64)

Metrics (profile Pearson, Jensen-Shannon distance, log-counts Pearson, counts Spearman) are saved to `performance_metrics/chrombpnet/<pseudobulk>.json`.

### 7. Attribution

First, filter peaks to remove any containing non-ACGT characters, then save one-hot-encoded sequences (run from `src/chrombpnet/attribute/`):

```bash
for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    python filter_nonACGT_regions.py -v \
        -f ../../../data/Sbicolor_730_v5.0.fa \
        -b ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
        -o ../../../data/peaks/${pseudobulk}_peaks_ACGT.narrowPeak
done

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    python save_ohe.py -v -p ${pseudobulk}
done
```

Then compute DeepLIFT/SHAP hypothetical attributions on a SLURM cluster:

```bash
export REPO_ROOT=/path/to/plant-design
sbatch src/chrombpnet/attribute/launch.sh
```

`launch.sh` submits an 8-task array (4 pseudobulks × 2 heads: profile, count) for the accessibility model. Attributions are saved to `attributions/{pseudobulk}_{model_type}_{head}.npz`, and one-hot-encoded sequences to `attributions/{pseudobulk}_ohe.npz`.
