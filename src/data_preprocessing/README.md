# Data Preprocessing

Run commands from the repository root. Python dependencies are managed by `uv`,
so use `uv run python` for Python entrypoints.

## Inputs

Shared inputs:

```text
data/Sbicolor_730_v5.0.fa
data/Sbicolor_730_v5.0.chrom.sizes
configs/pseudobulks.txt
configs/pseudobulks_coarse.txt
configs/pseudobulks_fine.txt
```

Dataset-specific peak and signal directories:

```text
data/peaks/
data/peaks_coarse/
data/peaks_fine/
data/bigwigs/
data/bigwigs_coarse/
data/bigwigs_fine/
```

## BigWigs

`make_bigwigs.sh` converts pseudobulk fragment alignments into shifted bigWigs.
It is cluster/path-specific, so inspect the source paths before running it on a
new filesystem.

```bash
bash src/data_preprocessing/make_bigwigs.sh
```

## Differential Accessibility

Build the union peak set, extract bigWig counts, normalize to CPM, and write DA
peak sets for downstream benchmarking, finetuning, and attribution.

```bash
uv run python src/data_preprocessing/differential_accessibility.py
uv run python src/data_preprocessing/differential_accessibility.py --dataset coarse
uv run python src/data_preprocessing/differential_accessibility.py --dataset fine
```

Outputs are written inside the selected peaks directory:

```text
data/peaks/differential_accessibility/
data/peaks_coarse/differential_accessibility/
data/peaks_fine/differential_accessibility/
```

Main outputs:

```text
da_peaks.csv
da_peaks_tau.bed
da_peaks_pairwise.bed
da_peaks_pairwise_higher.bed
<pseudobulk>_ranked.csv
summary.json
```

Use TMM normalization instead of CPM scaling with `--tmm`:

```bash
uv run python src/data_preprocessing/differential_accessibility.py --dataset coarse --tmm
```

## Pairwise-Higher BEDs

`differential_accessibility.py` writes the standard DA BED sets automatically.
If you need to regenerate only the pairwise-higher files from an existing
`da_peaks.csv`, run:

```bash
uv run python src/data_preprocessing/write_pairwise_higher_beds.py \
    --da data/peaks/differential_accessibility/da_peaks.csv \
    --out-dir data/peaks/differential_accessibility \
    --pseudobulks configs/pseudobulks.txt
```

For coarse or fine:

```bash
dataset=coarse
uv run python src/data_preprocessing/write_pairwise_higher_beds.py \
    --da "data/peaks_${dataset}/differential_accessibility/da_peaks.csv" \
    --out-dir "data/peaks_${dataset}/differential_accessibility" \
    --pseudobulks "configs/pseudobulks_${dataset}.txt"
```
