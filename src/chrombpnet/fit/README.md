# ChromBPNet Training

These launchers generate nonpeak regions, train bias models, and train
accessibility models for base, coarse, and fine pseudobulk datasets.

Training itself runs inside the ChromBPNet Apptainer image. Set
`CHROMBPNET_APPTAINER` to the image path or pass `--apptainer-image`.

## Inputs

```text
data/Sbicolor_730_v5.0.fa
data/Sbicolor_730_v5.0.chrom.sizes
configs/fold_0.json ... configs/fold_4.json
configs/pseudobulks.txt
configs/pseudobulks_coarse.txt
configs/pseudobulks_fine.txt
data/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/bigwigs{_dataset}/<pseudobulk>.bw
```

For `--dataset base`, paths omit the dataset suffix:

```text
data/peaks/
data/bigwigs/
data/nonpeaks/
models/
```

For `--dataset coarse` or `--dataset fine`, paths include the dataset namespace:

```text
data/peaks_coarse/
data/bigwigs_coarse/
data/nonpeaks_coarse/
models/coarse/
```

## Generate Nonpeaks

Submit all base-dataset nonpeak jobs:

```bash
python src/chrombpnet/fit/nonpeaks.py
```

Print local commands instead of submitting SLURM jobs:

```bash
python src/chrombpnet/fit/nonpeaks.py --interactive | parallel
```

Run a single coarse fold:

```bash
python src/chrombpnet/fit/nonpeaks.py \
    --dataset coarse \
    -p Epidermis_II \
    -f 0
```

The default negative-to-positive ratio is `-npr 1`.

## Train Bias Models

Bias jobs use the generated nonpeaks and write to `models{dataset}/`.

```bash
python src/chrombpnet/fit/bias.py
python src/chrombpnet/fit/bias.py --dataset coarse
python src/chrombpnet/fit/bias.py --dataset fine -p bundle_sheath -f 0
```

Use `--interactive` to print Apptainer commands that can be piped to your own
scheduler:

```bash
python src/chrombpnet/fit/bias.py --dataset coarse --interactive | parallel -j 4
```

Useful bias-only knobs:

```bash
python src/chrombpnet/fit/bias.py \
    --outlier-threshold 0.9999 \
    --bias-threshold-factor 0.5
```

## Train Accessibility Models

Accessibility training expects the matching bias model for each pseudobulk/fold
to already exist.

```bash
python src/chrombpnet/fit/accessibility.py
python src/chrombpnet/fit/accessibility.py --dataset coarse
python src/chrombpnet/fit/accessibility.py --dataset fine -p bundle_sheath -f 0
```

Interactive mode prints one command per missing job:

```bash
python src/chrombpnet/fit/accessibility.py --dataset fine --interactive | parallel -j 4
```

## Convert to PyTorch

After TensorFlow training finishes, convert the saved models for downstream
PyTorch analyses:

```bash
uv run python src/chrombpnet/fit/convert_to_torch.py
```

Use `--help` on each launcher for SLURM resource flags, including
`--cpus-per-task`, `--gpus`, `--time`, `--mem`, `--dry-run`, and `--rerun`.
