# Cherimoya Attribution

Run these commands from the repository root after Cherimoya models have been
trained with `src/cherimoya_pipeline/fit/launch.py`.

Python dependencies are managed by `uv`, so invoke Python entrypoints with
`uv run python`.

## Inputs

Required shared inputs:

```text
data/S.bicolor/Sbicolor_730_v5.0.fa
configs/S.bicolor/pseudobulks.txt
configs/S.bicolor/pseudobulks_fine.txt
configs/S.bicolor/pseudobulks_coarse.txt
configs/S.bicolor/chrom_splits.json
```

For all-peak attributions, each dataset also needs filtered peak files:

```text
data/S.bicolor/peaks/<pseudobulk>_peaks_ACGT.narrowPeak
data/S.bicolor/peaks_fine/<pseudobulk>_peaks_ACGT.narrowPeak
data/S.bicolor/peaks_coarse/<pseudobulk>_peaks_ACGT.narrowPeak
```

For DA-peak attributions, run differential accessibility first so these files
exist under `data/S.bicolor/peaks/differential_accessibility/`.

## Filter Peaks

Filter all-peak BEDs to windows with only A/C/G/T bases.

```bash
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/cherimoya_pipeline/attribute/filter_nonACGT_regions.py -v \
        -f data/S.bicolor/Sbicolor_730_v5.0.fa \
        -b "data/S.bicolor/peaks/${pseudobulk}_peaks.narrowPeak" \
        -o "data/S.bicolor/peaks/${pseudobulk}_peaks_ACGT.narrowPeak"
done < configs/S.bicolor/pseudobulks.txt
```

For fine or coarse datasets, change both the config and peak directory:

```bash
dataset=fine
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/cherimoya_pipeline/attribute/filter_nonACGT_regions.py -v \
        -f data/S.bicolor/Sbicolor_730_v5.0.fa \
        -b "data/S.bicolor/peaks_${dataset}/${pseudobulk}_peaks.narrowPeak" \
        -o "data/S.bicolor/peaks_${dataset}/${pseudobulk}_peaks_ACGT.narrowPeak"
done < "configs/S.bicolor/pseudobulks_${dataset}.txt"
```

To run the filter jobs in parallel, print commands and pipe them:

```bash
dataset=base
config="configs/S.bicolor/pseudobulks.txt"
peaks_dir="data/S.bicolor/peaks"

while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    printf '%q ' uv run python src/cherimoya_pipeline/attribute/filter_nonACGT_regions.py -v \
        -f data/S.bicolor/Sbicolor_730_v5.0.fa \
        -b "${peaks_dir}/${pseudobulk}_peaks.narrowPeak" \
        -o "${peaks_dir}/${pseudobulk}_peaks_ACGT.narrowPeak"
    printf '\n'
done < "${config}" | parallel
```

For coarse/fine datasets:

```bash
dataset=coarse
config="configs/S.bicolor/pseudobulks_${dataset}.txt"
peaks_dir="data/S.bicolor/peaks_${dataset}"

while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    printf '%q ' uv run python src/cherimoya_pipeline/attribute/filter_nonACGT_regions.py -v \
        -f data/S.bicolor/Sbicolor_730_v5.0.fa \
        -b "${peaks_dir}/${pseudobulk}_peaks.narrowPeak" \
        -o "${peaks_dir}/${pseudobulk}_peaks_ACGT.narrowPeak"
    printf '\n'
done < "${config}" | parallel -j 4
```

## Save One-Hot Encoded Sequences

Save one-hot encoded sequence arrays once per pseudobulk before attribution:

```bash
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/cherimoya_pipeline/attribute/save_ohe.py -v -p "${pseudobulk}"
done < configs/S.bicolor/pseudobulks.txt
```

For fine or coarse datasets:

```bash
dataset=coarse
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/cherimoya_pipeline/attribute/save_ohe.py -v \
        --dataset "${dataset}" \
        -p "${pseudobulk}"
done < "configs/S.bicolor/pseudobulks_${dataset}.txt"
```

For base-dataset DA peak attributions, create the DA BEDs first:

```bash
uv run python src/data_preprocessing/differential_accessibility.py
uv run python src/data_preprocessing/write_pairwise_higher_beds.py
```

Then save OHE for the DA loci. `specific_union` is the per-cell-type union used
for per-pseudobulk specificity analyses.

```bash
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/cherimoya_pipeline/attribute/save_ohe.py -v \
        -p "${pseudobulk}" \
        --da-peaks specific_union
done < configs/S.bicolor/pseudobulks.txt
```

Other supported DA peak sets are `tau`, `pairwise`, and `pairwise_higher`.

## Launch DeepLIFT/SHAP Jobs

`launch.py` submits one SLURM job per (pseudobulk, head) pair whose output
is missing, running via `apptainer exec --nv` against the Cherimoya
Apptainer image (see `../apptainer/`) by default. `--dry-run` prints the
generated sbatch scripts instead of submitting; `--local` runs in the
foreground instead of submitting to SLURM.

All base all-peak attributions:

```bash
uv run python src/cherimoya_pipeline/attribute/launch.py
uv run python src/cherimoya_pipeline/attribute/launch.py --dry-run
```

Fine or coarse all-peak attributions:

```bash
uv run python src/cherimoya_pipeline/attribute/launch.py --dataset coarse
```

Base DA-specific attribution jobs:

```bash
uv run python src/cherimoya_pipeline/attribute/launch.py \
    --da-peaks specific_union
```

Finetuned and adapter DA-specific runs:

```bash
uv run python src/cherimoya_pipeline/attribute/launch.py \
    --da-peaks specific_union \
    --finetuned

uv run python src/cherimoya_pipeline/attribute/launch.py \
    --da-peaks specific_union \
    --finetuned \
    --da-fraction 0.8

uv run python src/cherimoya_pipeline/attribute/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.5

uv run python src/cherimoya_pipeline/attribute/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.8
```
