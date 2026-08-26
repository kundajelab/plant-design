# Data Preprocessing

Run commands from the repository root. Python dependencies are managed by `uv`,
so use `uv run python` for Python entrypoints.

## Inputs

Shared inputs (`--species`/`$SPECIES` defaults to `S.bicolor`):

```text
data/<species>/<genome>.fa
data/<species>/<genome>.chrom.sizes
configs/<species>/pseudobulks.txt
configs/<species>/pseudobulks_coarse.txt
configs/<species>/pseudobulks_fine.txt
```

Dataset-specific peak and signal directories:

```text
data/<species>/peaks/
data/<species>/peaks_coarse/
data/<species>/peaks_fine/
data/<species>/bigwigs/
data/<species>/bigwigs_coarse/
data/<species>/bigwigs_fine/
```

## BigWigs

`make_bigwigs.sh` resolves the right paths (genome/chrom-sizes, fragments
source, output dir, pseudobulk list) once, then prints one `bam2bw -f` command
per pseudobulk in `--dataset` to stdout -- like the other launchers in this
repo, it doesn't run them itself. Pipe into `parallel`/`simple_gpu_scheduler`
to actually run the jobs. Fragments are read straight from `DATA_DIR` on Oak
(the whole pipeline runs on Oak, so there's no local fragments copy/cache
step). `$SPECIES` selects the namespace (default `S.bicolor`); `$DATASET`
selects `base`/`fine`/`coarse` (default `base`) and switches the Oak source
path, the `configs/<species>/pseudobulks{_fine,_coarse}.txt` pseudobulk list,
and the `data/<species>/bigwigs{_fine,_coarse}/` output dir together. Other
knobs (`DATA_DIR`, `REF_DIR`, `OUT_DIR`, `FRAGMENTS_SUFFIX`,
`PSEUDOBULKS_FILE`) override the per-dataset defaults. `GENOME_FASTA_NAME`
defaults to whatever `configs/<species>/species.json` names (the same
manifest `species_config.genome_fasta()` reads) rather than to Sorghum's
filename, so other species don't silently look for a genome that was never
fetched under their own `data/<species>/`. The genome rsync/faidx/chrom-sizes
steps are skipped automatically if the genome is already present, so species
whose genome was fetched a different way (see "Arabidopsis" below) don't
need a working `REF_DIR`.

```bash
bash src/data_preprocessing/make_bigwigs.sh | parallel -j 8
DATASET=fine bash src/data_preprocessing/make_bigwigs.sh | parallel -j 8
DATASET=coarse bash src/data_preprocessing/make_bigwigs.sh | parallel -j 8
```

### Arabidopsis (A.thaliana)

The `MoralesArabidopsis` pseudobulk dataset lives at
`/oak/stanford/groups/akundaje/ayhe/plant-design/data/A.thaliana/` with
`peaks/` and `fragments/` subdirectories; `configs/A.thaliana/pseudobulks.txt`
lists its 30 cell types. Its fragment files are named
`<pseudobulk>.fragments.tsv.gz` (dot separator), unlike Sorghum's
`<pseudobulk>_fragments.tsv.gz`, so `FRAGMENTS_SUFFIX` must be overridden.

Fetch the reference genome once (classic TAIR10 -- see
`download_arabidopsis_genome.sh` for why this specific assembly, not the
2023 RefSeq re-annotation, matches this dataset's chromosome sizes):

```bash
bash src/data_preprocessing/download_arabidopsis_genome.sh
```

Copy peaks and build bigWigs:

```bash
## REPLACE THIS WHEN PROJECT GOES PUBLIC WITH DIRECT DOWNLOADS FROM ZENODO/ETC
SPECIES=A.thaliana \
DATASET=base \
DATA_DIR=/oak/stanford/groups/akundaje/ayhe/plant-design/data/A.thaliana/peaks \
bash src/data_preprocessing/cp_peaks.sh

SPECIES=A.thaliana \
DATA_DIR=/oak/stanford/groups/akundaje/ayhe/plant-design/data/A.thaliana/fragments \
FRAGMENTS_SUFFIX=.fragments.tsv.gz \
bash src/data_preprocessing/make_bigwigs.sh | parallel -j 8
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

For our `coarse` analyses, we use `--tmm --tau-threshold 0.7`; for the baseline models, we used `--tmm --tau-threshold 0.75`.

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
