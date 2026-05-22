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
mamba create -n torch python=3.14
mamba activate torch
pip install bpnet-lite tangermeme tqdm bam2bw ledidi daicon
```

`daicon` provides the low-rank convolutional/linear adapter implementation used
for differential accessibility fine-tuning. Source:
<https://github.com/adamyhe/daicon>.

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
python src/chrombpnet/benchmark/benchmark.py -p <pseudobulk> --adapter --da-fraction 0.8
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

Benchmark a trained adapter sweep:

```bash
python src/chrombpnet/benchmark/benchmark_lora_sweep.py \
    --sweep-id <sweep_id> \
    --dry-run

python src/chrombpnet/benchmark/benchmark_lora_sweep.py \
    --sweep-id <sweep_id>
```

Summarize and plot completed adapter benchmarks, including base and full-model
fine-tuned references:

```bash
python src/chrombpnet/benchmark/plot_lora_sweep.py \
    --sweep-id <sweep_id>
```

Pick the best LoRA adapter model from the summary CSV:

```bash
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
export REPO_ROOT=/path/to/plant-design
sbatch src/chrombpnet/attribute/launch.sh
```

`launch.sh` submits an 8-task array (4 pseudobulks × 2 heads: profile, count) for the accessibility model. Attributions are saved to `attributions/{pseudobulk}_{model_type}_{head}.npz`, and one-hot-encoded sequences to `attributions/{pseudobulk}_ohe.npz`.

Adapter attribution is supported by passing the trained adapter sweep ID through
the attribution launcher or DeepLIFT script:

```bash
python src/chrombpnet/attribute/deepliftshap.py \
    -p <pseudobulk> \
    -m accessibility \
    --head profile \
    --adapter-sweep-id <sweep_id>
```

`daicon` adapters are compatible with tangermeme DeepLIFT/SHAP in evaluation
mode because the adapter branches are affine residual additions over frozen
`Conv1d`/`Linear` layers. Dropout may be used during training but is disabled
with `model.eval()` during attribution.
