# ChromBPNet Attribution

Run these commands from the repository root after ChromBPNet models have been
converted to PyTorch with `src/chrombpnet/fit/convert_to_torch.py`.

Python dependencies are managed by `uv`, so invoke Python entrypoints with
`uv run python`.

## Inputs

Required shared inputs:

```text
data/Sbicolor_730_v5.0.fa
configs/pseudobulks.txt
configs/pseudobulks_fine.txt
configs/pseudobulks_coarse.txt
configs/chrom_splits.json
```

For all-peak attributions, each dataset also needs filtered peak files:

```text
data/peaks/<pseudobulk>_peaks_ACGT.narrowPeak
data/peaks_fine/<pseudobulk>_peaks_ACGT.narrowPeak
data/peaks_coarse/<pseudobulk>_peaks_ACGT.narrowPeak
```

For DA-peak attributions, run differential accessibility first so these files
exist under `data/peaks/differential_accessibility/`.

## Filter Peaks

Filter all-peak BEDs to windows with only A/C/G/T bases.

```bash
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/chrombpnet/attribute/filter_nonACGT_regions.py -v \
        -f data/Sbicolor_730_v5.0.fa \
        -b "data/peaks/${pseudobulk}_peaks.narrowPeak" \
        -o "data/peaks/${pseudobulk}_peaks_ACGT.narrowPeak"
done < configs/pseudobulks.txt
```

For fine or coarse datasets, change both the config and peak directory:

```bash
dataset=fine
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/chrombpnet/attribute/filter_nonACGT_regions.py -v \
        -f data/Sbicolor_730_v5.0.fa \
        -b "data/peaks_${dataset}/${pseudobulk}_peaks.narrowPeak" \
        -o "data/peaks_${dataset}/${pseudobulk}_peaks_ACGT.narrowPeak"
done < "configs/pseudobulks_${dataset}.txt"
```

To run the filter jobs in parallel, print commands and pipe them:

```bash
dataset=base
config="configs/pseudobulks.txt"
peaks_dir="data/peaks"

while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    printf '%q ' uv run python src/chrombpnet/attribute/filter_nonACGT_regions.py -v \
        -f data/Sbicolor_730_v5.0.fa \
        -b "${peaks_dir}/${pseudobulk}_peaks.narrowPeak" \
        -o "${peaks_dir}/${pseudobulk}_peaks_ACGT.narrowPeak"
    printf '\n'
done < "${config}" | parallel
```

For coarse/fine datasets:

```bash
dataset=coarse
config="configs/pseudobulks_${dataset}.txt"
peaks_dir="data/peaks_${dataset}"

while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    printf '%q ' uv run python src/chrombpnet/attribute/filter_nonACGT_regions.py -v \
        -f data/Sbicolor_730_v5.0.fa \
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
    uv run python src/chrombpnet/attribute/save_ohe.py -v -p "${pseudobulk}"
done < configs/pseudobulks.txt
```

For fine or coarse datasets:

```bash
dataset=coarse
while read -r pseudobulk; do
    test -n "${pseudobulk}" || continue
    uv run python src/chrombpnet/attribute/save_ohe.py -v \
        --dataset "${dataset}" \
        -p "${pseudobulk}"
done < "configs/pseudobulks_${dataset}.txt"
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
    uv run python src/chrombpnet/attribute/save_ohe.py -v \
        -p "${pseudobulk}" \
        --da-peaks specific_union
done < configs/pseudobulks.txt
```

Other supported DA peak sets are `tau`, `pairwise`, and `pairwise_higher`.

## Launch DeepLIFT/SHAP Jobs

All base all-peak attributions:

```bash
uv run python src/chrombpnet/attribute/launch.py --model accessibility
```

Fine or coarse all-peak attributions:

```bash
uv run python src/chrombpnet/attribute/launch.py \
    --dataset coarse \
    --model accessibility \
    --time 24:00:00
```

Base DA-specific attribution jobs:

```bash
uv run python src/chrombpnet/attribute/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --time 6:00:00
```

Finetuned and adapter DA-specific runs:

```bash
uv run python src/chrombpnet/attribute/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --time 6:00:00

uv run python src/chrombpnet/attribute/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --da-fraction 0.8 \
    --time 6:00:00

uv run python src/chrombpnet/attribute/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.5 \
    --time 6:00:00

uv run python src/chrombpnet/attribute/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.8 \
    --time 6:00:00
```

Use `--dry-run` to inspect SLURM scripts before submitting.
