# Low-Rank Adapter Handoff

This document describes the ChromBPNet low-rank adapter implementation in this
repo so it can be ported into an independent codebase. The implementation is
PyTorch-only and targets bpnet-lite `ChromBPNet` models whose accessibility
submodel is a bpnet-lite `BPNet`.

## Goal

Fine-tune cell-type-specific differential accessibility behavior without
modifying the base ChromBPNet weights. The base `bias` model remains frozen, the
base `accessibility` model remains frozen, and small trainable residual adapters
are attached to selected accessibility layers. Adapter weights are saved as
separate artifacts and can be plugged into the matching base fold model for
prediction, benchmarking, and DeepLIFT/SHAP attribution.

## Core Architecture

The adapter code lives in:

- `src/chrombpnet/adapters/layers.py`
- `src/chrombpnet/adapters/utils.py`

Two adapter wrappers are implemented:

- `LowRankConv1dAdapter`
  - Wraps a frozen `torch.nn.Conv1d`.
  - Computes `base(x) + scale * up(dropout(down(x)))`.
  - `down` is a low-rank `Conv1d` using the same kernel, stride, padding,
    dilation, groups, and padding mode as the base convolution.
  - `up` is a `1x1` convolution from `rank -> out_channels`.

- `LowRankLinearAdapter`
  - Wraps a frozen `torch.nn.Linear`.
  - Computes `base(x) + scale * up(dropout(down(x)))`.
  - `down` maps `in_features -> rank`.
  - `up` maps `rank -> out_features`.

The `up` layer is initialized to zero, so an injected adapter is exactly
prediction-equivalent to the base model before training. Base layer parameters
are frozen inside the wrapper.

The scale convention is:

```python
conv_scale = conv_alpha / conv_rank
linear_scale = linear_alpha / linear_rank
```

With `alpha=1`, rank 1 has scale `1.0`, rank 4 has scale `0.25`, and rank 16
has scale `0.0625`.

## bpnet-lite Injection Targets

The implementation assumes the accessibility model has bpnet-lite-style fields:

- `accessibility.iconv`
- `accessibility.rconvs`
- `accessibility.fconv`
- `accessibility.linear`

Supported target names:

- `iconv`: first sequence convolution
- `rconvs`: residual/dilated convolution stack
- `fconv`: profile head convolution
- `linear`: count head

Supported presets:

- `profile_only`: `fconv`
- `count_only`: `linear`
- `body_only`: `iconv`, `rconvs`
- `heads_only`: `fconv`, `linear`
- `all_conv`: `iconv`, `rconvs`, `fconv`
- `all`: `iconv`, `rconvs`, `fconv`, `linear`
- `rconvs+heads`: `rconvs`, `fconv`, `linear`

The main public utility is:

```python
inject_adapters(
    accessibility,
    targets,
    conv_rank=4,
    linear_rank=4,
    conv_scale=None,
    linear_scale=None,
    dropout=0.0,
)
```

It freezes the accessibility model, wraps requested layers, and returns a config
dictionary containing resolved targets, ranks, scales, dropout, injected layer
names, and layer specs.

## Artifact Contract

Adapters are saved separately from the base model:

```text
models/{pseudobulk}/adapters/{sweep_id}/{pseudobulk}_f{fold}.adapter.pt
models/{pseudobulk}/adapters/{sweep_id}/{pseudobulk}_f{fold}.config.json
models/{pseudobulk}/adapters/{sweep_id}/{pseudobulk}_f{fold}.log.tsv
```

The `.adapter.pt` file is a `torch.save` bundle with:

```python
{
    "manifest": {...},
    "state_dict": adapter_only_state_dict,
}
```

The state dict excludes frozen `base.*` weights and contains only adapter branch
weights (`down.*`, `up.*`) keyed by wrapped module path.

The manifest includes:

- `format_version`
- `pseudobulk`
- `fold`
- `base_model_path`
- `sweep_id`
- `adapter_config`
  - targets
  - conv rank
  - linear rank
  - conv scale
  - linear scale
  - dropout
- `layer_specs`
- `trainable_params`
- training hyperparameters

For job completion checks, use the final `.log.tsv` as the finish artifact. The
adapter bundle can be written before the loop finishes because the best
checkpoint is saved during training.

## Loading Contract

There are two loader helpers:

```python
load_model_with_adapter(base_model_path, adapter_path=None, map_location="cpu")
load_accessibility_with_adapter(base_accessibility_path, adapter_path=None, map_location="cpu")
```

The first loads a full ChromBPNet model and injects adapters into
`model.accessibility`. The second loads only an accessibility BPNet and injects
adapters into it for attribution workflows.

Important behavior:

- The adapter manifest is used to reconstruct the adapter structure.
- Manifest layer specs are checked against the injected layers.
- Adapter weights are loaded strictly.
- The injected model is moved back to the same device as the loaded base model,
  preventing CUDA input vs CPU adapter-weight mismatches.

## Training Flow

Adapter training lives in:

- `src/chrombpnet/da_finetuning/finetune_adapters.py`
- It reuses dataset and extraction helpers from `finetune_da.py`.

High-level flow:

1. Load DA peaks from:
   - `data/peaks/differential_accessibility/da_peaks_tau.bed`
   - `data/peaks/differential_accessibility/da_peaks_pairwise.bed`
2. Load cell-type peaks and fold chromosome splits.
3. Load base full ChromBPNet model:
   - `models/{pseudobulk}/torch/{pseudobulk}_f{fold}.torch`
4. Freeze all base parameters.
5. Inject adapters into `model.accessibility`.
6. Move the model to the selected device after adapter injection.
7. Build the mixed DA fine-tuning dataset:
   - DA peaks with jitter and reverse complement augmentation.
   - Base mix of cell-type peaks and nonpeaks.
8. Train only adapter parameters.
9. Validate on held-out fold DA peaks.
10. Save best adapter bundle during training.
11. Write final `.log.tsv` at the end of the training loop.

The loss matches the full fine-tuning path:

```python
loss = MNLLLoss(profile_log_probs, observed_profile).mean()
     + alpha * log1pMSELoss(predicted_log_counts, observed_log_counts).mean()
```

The forward pass uses the full ChromBPNet object so the frozen bias model is
combined exactly as in normal prediction.

## Training CLI

Example:

```bash
python src/chrombpnet/da_finetuning/finetune_adapters.py \
    -p mesophyll \
    -f 0 \
    --sweep-id adapter_t-heads_only_cr4_lr4_a4_d0_lr0p0003_da0p5 \
    --targets heads_only \
    --conv-rank 4 \
    --linear-rank 4 \
    --conv-alpha 4 \
    --linear-alpha 4 \
    --dropout 0.0 \
    --lr 0.0003 \
    --da-fraction 0.5 \
    --epochs 100 \
    --patience 10 \
    -v
```

## Sweep Launcher

Adapter training sweep launching lives in:

- `src/chrombpnet/da_finetuning/launch_adapter_sweep.py`

It expands a JSON sweep config into `finetune_adapters.py` commands. It submits
one SLURM job per `(pseudobulk, fold)` and can also print commands with
`--interactive`. Completion checks use `.log.tsv`, not `.adapter.pt`.

Example smoke config:

- `configs/adapter_sweeps/smoke.json`

Typical status check:

```bash
python src/chrombpnet/da_finetuning/launch_adapter_sweep.py \
    --config configs/adapter_sweeps/smoke.json \
    --check-finished
```

## Benchmarking and Model Selection

Adapter-aware benchmarking was added to:

- `src/chrombpnet/benchmark/benchmark.py`
- `src/chrombpnet/benchmark/benchmark_da.py`
- `src/chrombpnet/benchmark/pairwise_correlations_predicted.py`
- `src/chrombpnet/benchmark/plot_celltype_scatter.py`

They all accept:

```bash
--adapter-sweep-id <sweep_id>
```

Benchmark launcher:

- `src/chrombpnet/benchmark/benchmark_lora_sweep.py`

This submits single-GPU serial SLURM jobs for missing benchmark outputs. It can
include base and full-model fine-tuned references. It does not plot.

Plotting and summary:

- `src/chrombpnet/benchmark/plot_lora_sweep.py`

This reads completed JSON/CSV outputs and writes:

```text
performance_metrics/plots/lora_sweep/lora_sweep_summary.csv
performance_metrics/plots/lora_sweep/lora_cross_loci_metrics.pdf
performance_metrics/plots/lora_sweep/lora_da_metrics.pdf
performance_metrics/plots/lora_sweep/lora_pairwise_count_metrics.pdf
```

Model selection:

- `src/chrombpnet/benchmark/select_best_lora_model.py`

It ranks LoRA adapter candidates from `lora_sweep_summary.csv` using weighted,
z-scored metrics. Defaults emphasize DA performance:

- `specificity_accuracy_pairwise_filtered`: `2.0`
- `mean_pairwise_pearson_filtered`: `2.0`
- `mean_log2fc_pearson_tau_filtered`: `1.0`
- `tau_auprc_all_peaks`: `1.0`
- `log_counts_pearson`: `0.5`
- `profile_pearson`: `0.5`
- `profile_jsd`: `-0.5`

## DeepLIFT/SHAP Compatibility

Tangermeme's DeepLIFT/SHAP implementation hooks known nonlinear modules and
treats other operations as linear/affine. The adapter wrappers are compatible
because, in evaluation mode, they contain only:

- frozen `Conv1d` or `Linear`
- trainable `Conv1d` or `Linear`
- addition and scalar multiplication

Dropout can be used during training, but must be disabled during attribution.
The helper:

```python
assert_deeplift_shap_compatible(module)
```

checks that adapter dropout is inactive and no unexpected child modules were
added. The attribution path calls `model.eval()` before this check.

## Tests

Tests live in:

- `tests/test_adapters.py`

They cover:

- Conv adapter identity at initialization.
- Linear adapter identity at initialization.
- Adapter branches are affine in eval mode.
- Dropout must be disabled before DeepLIFT/SHAP compatibility.
- Only adapter branch parameters require gradients.
- Adapter save/load reproduces predictions.
- Manifest mismatch fails.
- Loading adapter bundles preserves the base model device.

Run:

```bash
pytest -q tests/test_adapters.py
```

## Suggested Independent Repo Layout

```text
chrombpnet_lora/
  chrombpnet_lora/
    __init__.py
    layers.py
    inject.py
    io.py
    train.py
    data.py
    benchmark.py
    attribution.py
  configs/
    smoke.json
  scripts/
    finetune_adapters.py
    launch_adapter_sweep.py
    benchmark_lora_sweep.py
    plot_lora_sweep.py
    select_best_model.py
  tests/
    test_adapters.py
```

Recommended separations:

- Keep layer wrappers independent of ChromBPNet.
- Keep bpnet-lite-specific target resolution in a bpnet-lite adapter module.
- Keep training data extraction project-specific.
- Keep the adapter artifact format stable and versioned.
- Treat base model paths as external inputs instead of assuming this repo's
  `models/{pseudobulk}/torch/` layout.

## Porting Checklist

1. Copy `layers.py` and adapter IO/injection helpers.
2. Replace hard-coded repo paths with CLI/config inputs.
3. Define a model adapter interface:
   - load base model
   - locate accessibility submodel
   - inject adapters
   - run forward pass
4. Keep `.adapter.pt`, `.config.json`, and `.log.tsv` as explicit artifacts.
5. Use `.log.tsv` for completion checks.
6. Add integration tests with a tiny model before using real ChromBPNet models.
7. Verify DeepLIFT/SHAP compatibility with `model.eval()` and the compatibility
   assertion.
