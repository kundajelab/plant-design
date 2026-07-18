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
mamba env create -f environment.yml
mamba activate plant-design
uv sync --frozen
```

Python dependencies for preprocessing, conversion, benchmarking, attribution,
sequence design, and adapter fine-tuning are managed in `pyproject.toml` and
locked in `uv.lock`. Run PyTorch-side entrypoints through `uv run`, for example
`uv run python src/chrombpnet/fit/convert_to_torch.py`.

## Pipeline

### 1. Data preprocessing

Convert ATAC fragment files to per-pseudobulk bigwigs:

```bash
bash src/data_preprocessing/make_bigwigs.sh
bash src/data_preprocessing/make_bigwigs_fine.sh
bash src/data_preprocessing/make_bigwigs_coarse.sh
```

Expected data layout under `data/`:
```
data/
├── Sbicolor_730_v5.0.fa          # Sorghum v5 reference genome
├── peaks/<pseudobulk>_peaks.narrowPeak
├── peaks_fine/<pseudobulk>_peaks.narrowPeak
├── peaks_coarse/<pseudobulk>_peaks.narrowPeak
├── bigwigs/<pseudobulk>.bw
├── bigwigs_fine/<pseudobulk>.bw
└── bigwigs_coarse/<pseudobulk>.bw
```

The base dataset is the default for training and analysis scripts. Use
`--dataset fine` or `--dataset coarse` with the training launchers and Python
entrypoints such as `convert_to_torch.py`, `benchmark.py`, and DeepLIFT/SHAP
scripts.

### 2. Generate non-peak background regions

```bash
bash src/chrombpnet/fit/nonpeaks.sh
```

### 3. Train bias models

Trains Tn5 insertion bias models for all pseudobulks across all 5 folds:

```bash
bash src/chrombpnet/fit/bias.sh
bash src/chrombpnet/fit/bias.sh --dataset fine --dry-run
```

### 4. Train accessibility (ChromBPNet) models

```bash
bash src/chrombpnet/fit/accessibility.sh
bash src/chrombpnet/fit/accessibility.sh --dataset fine --dry-run
```

### 5. Convert to PyTorch

Converts trained TensorFlow `.h5` models to PyTorch `.torch` format:

```bash
uv run python src/chrombpnet/fit/convert_to_torch.py
uv run python src/chrombpnet/fit/convert_to_torch.py --dataset fine
```

### 6. Benchmark

Evaluates models on held-out test chromosomes per fold:

```bash
uv run python src/chrombpnet/benchmark/benchmark.py -p <pseudobulk> [-o]
uv run python src/chrombpnet/benchmark/benchmark.py -p <pseudobulk> --adapter --da-fraction 0.8
```

Options:
- `-p` / `--pseudobulk`: one of `bundle_sheath`, `epidermis`, `guard`, `mesophyll`, `phloem`, `xylem`
- `-o` / `--save-output`: save scaled predictions and signals to `predictions/chrombpnet/<pseudobulk>.npz`
- `-b` / `--batch-size`: inference batch size (default: 64)
- `--adapter`: benchmark the default daicon adapter run for the selected `--da-fraction`
- `--adapter-sweep-id` / `--adapter-run-id`: benchmark an explicit adapter run ID

Metrics (profile Pearson, Jensen-Shannon distance, log-counts Pearson, counts Spearman) are saved to `performance_metrics/chrombpnet/<pseudobulk>.json`.

### 7. Differential accessibility

Build the union peak set, extract observed accessibility counts, and compute DA
specificity metrics:

```bash
python src/data_preprocessing/differential_accessibility.py
```

Outputs are written under `data/peaks/differential_accessibility/` and are
required for DA benchmarking and adapter fine-tuning.

Evaluate whether out-of-fold model predictions recover observed cell-type
specificity:

```bash
python src/chrombpnet/benchmark/benchmark_da.py [--save-output]
python src/chrombpnet/benchmark/benchmark_da.py --adapter --da-fraction 0.8
```

### 8. Full-model DA fine-tuning

The original DA fine-tuning path updates the full accessibility submodel while
keeping the bias model frozen:

```bash
python src/chrombpnet/da_finetuning/finetune_da.py -p <pseudobulk> -f <fold> -v
```

Fine-tuned models are saved under `models/<pseudobulk>/finetuned/`.
Use `--da-fraction 0.8` to save the 80:20 DA replay run under
`models/<pseudobulk>/finetuned_0.8/`.

### 9. Low-rank adapter DA fine-tuning

Adapter fine-tuning freezes the base ChromBPNet model and trains only small
low-rank residual adapters from `daicon`. Adapter bundles are saved separately
and are loaded onto the matching base fold model at prediction/attribution time.
The frozen bias and accessibility base weights are not included in the adapter
state dict; only adapter branch weights are saved.

The main DA fine-tuning entrypoint supports daicon adapter mode. By default it
uses the daicon BPNet defaults selected from the parameter sweep: all BPNet
layers, `conv_rank=16`, `linear_rank=4`, dropout `0`, and alpha-8 scaling.

```bash
python src/chrombpnet/da_finetuning/finetune_da.py \
    -p <pseudobulk> -f <fold> --mode adapter --da-fraction 0.5 -v

python src/chrombpnet/da_finetuning/finetune_da.py \
    -p <pseudobulk> -f <fold> --mode adapter --da-fraction 0.8 -v
```

Artifacts:

```text
models/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.adapter.pt
models/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.config.json
models/<pseudobulk>/adapters/daicon_default_da0p5/<pseudobulk>_f<fold>.log.tsv
models/<pseudobulk>/adapters/daicon_default_da0p8/<pseudobulk>_f<fold>.adapter.pt
```

The `.adapter.pt` bundle contains a manifest plus adapter-only `state_dict`.
The manifest records the pseudobulk, fold, base model path, adapter config,
layer specs, trainable parameter count, and training hyperparameters. Use the
final `.log.tsv` as the completion marker because the best adapter checkpoint is
written during training before the loop finishes.

The legacy `finetune_adapters.py` CLI remains available as a compatibility
wrapper around `finetune_da.py --mode adapter`.

Use the standard launcher to submit full-model DA fine-tuning plus daicon
adapter runs. It submits one SLURM job per run group, pseudobulk, and fold,
with both 0.5 and 0.8 DA fractions by default. Adapter fine-tuning uses
`--adapter-lr 0.001` by default:

```bash
python src/chrombpnet/da_finetuning/launch.py --dry-run

python src/chrombpnet/da_finetuning/launch.py \
    --run full \
    --run adapter
```

Use the sweep launcher for the broad adapter hyperparameter grid. It submits one
SLURM job per pseudobulk/fold and uses the final `.log.tsv` as the completion
marker:

```bash
python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json \
    --dry-run

python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json
```

Check finished adapter jobs:

```bash
python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json \
    --check-finished
```

Benchmark the active model set (base, both full-finetuned models, and both
default daicon adapter models):

```bash
python src/chrombpnet/benchmark/launch.py --dry-run
python src/chrombpnet/benchmark/launch.py
```

Summarize and plot the active model benchmark outputs:

```bash
python src/chrombpnet/benchmark/plot_benchmark.py
```

The LoRA sweep benchmark and selection scripts are development-only utilities
for choosing adapter hyperparameters:

```bash
python src/chrombpnet/benchmark/benchmark_lora_sweep.py --sweep-id <sweep_id>
python src/chrombpnet/benchmark/plot_lora_sweep.py --sweep-id <sweep_id>
python src/chrombpnet/benchmark/select_best_lora_model.py \
    --summary performance_metrics/plots/lora_sweep/lora_sweep_summary.csv
```

### 10. Attribution

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
python src/chrombpnet/attribute/launch.py --model accessibility
```

`launch.py` submits one SLURM job per pseudobulk, model type, and head.
Attributions are saved under `attributions/chrombpnet/`, and one-hot-encoded
sequences to `attributions/chrombpnet/`.

Default daicon adapter attribution is supported through the attribution launcher:

```bash
python src/chrombpnet/attribute/launch.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.5

python src/chrombpnet/attribute/launch.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.8
```

Custom adapter sweep/run attribution is supported by passing the trained adapter
run ID through the attribution launcher or DeepLIFT script:

```bash
python src/chrombpnet/attribute/deepliftshap.py \
    -p <pseudobulk> \
    -m accessibility \
    --head profile \
    --adapter-run-id <adapter_id>
```

`daicon` adapters are compatible with tangermeme DeepLIFT/SHAP in evaluation
mode because the adapter branches are affine residual additions over frozen
`Conv1d`/`Linear` layers. Dropout may be used during training but is disabled
with `model.eval()` during attribution.

Run tf-modisco on attribution outputs and generate reports:

```bash
python src/chrombpnet/modisco/launch.py --model accessibility
python src/chrombpnet/modisco/launch_report.py --model accessibility
```

Default daicon adapter tf-modisco runs use the same adapter selectors:

```bash
python src/chrombpnet/modisco/launch.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.5

python src/chrombpnet/modisco/launch.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.8

python src/chrombpnet/modisco/launch_report.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.5

python src/chrombpnet/modisco/launch_report.py \
    --model accessibility \
    --adapter \
    --da-fraction 0.8
```
