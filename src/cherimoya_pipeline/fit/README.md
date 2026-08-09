# Cherimoya Training

These launchers generate GC-matched negative regions and train Cherimoya
accessibility models for base, coarse, and fine pseudobulk datasets.
Cherimoya trains a single model end-to-end directly on raw signal -- there
is no separate bias-model stage and no post-hoc PyTorch conversion step
(ChromBPNet's two-stage bias/accessibility training lives on the
`chrombpnet` branch).

`cherimoya` is a Linux-only core dependency of the `plant-design` project
(its `torch>=2.9`/`triton>=3.5.1` requirement has no macOS wheels), so
training and negative sampling both run via `uv run` on a Linux GPU
machine.

## Inputs

```text
data/<species>/<genome>.fa
data/<species>/<genome>.chrom.sizes
configs/<species>/fold_0.json ... fold_4.json
configs/<species>/pseudobulks.txt
configs/<species>/pseudobulks_coarse.txt
configs/<species>/pseudobulks_fine.txt
data/<species>/peaks{_dataset}/<pseudobulk>_peaks.narrowPeak
data/<species>/bigwigs{_dataset}/<pseudobulk>.bw
```

`--species` defaults to `S.bicolor` (see `src/species_config.py`).

For `--dataset base`, paths omit the dataset suffix (`data/<species>/peaks/`,
`.../bigwigs/`, `.../negatives/`, `models/<species>/`). For `--dataset coarse`
or `--dataset fine`, paths include the dataset namespace
(`data/<species>/peaks_coarse/`, `models/<species>/coarse/`, ...).

## Generate Negatives

Negatives are sampled once per pseudobulk (not per fold) -- Cherimoya's own
`negatives` CLI has no chromosome-split concept, and training-time chromosome
filtering already restricts both peaks and negatives to the active fold.

```bash
uv run python src/cherimoya_pipeline/fit/negatives.py
uv run python src/cherimoya_pipeline/fit/negatives.py --species A.thaliana | parallel
```

## Train

No SLURM integration: `launch.py` collects pseudobulk/fold jobs with a
missing model checkpoint and prints one command per job to stdout.

```bash
uv run python src/cherimoya_pipeline/fit/launch.py | parallel -j 4
uv run python src/cherimoya_pipeline/fit/launch.py --dataset coarse
uv run python src/cherimoya_pipeline/fit/launch.py -p bundle_sheath -f 0
```

Useful training knobs (see `train_cherimoya.py --help` for the full list):

```bash
uv run python src/cherimoya_pipeline/fit/launch.py \
    --n-filters 128 \
    --n-layers 9 \
    --max-epochs 50 \
    --early-stopping 5 \
    --negative-ratio 0.25
```

`train_cherimoya.py` is the actual training entrypoint; run it directly for
local debugging on a Linux GPU machine. It resolves our species/fold paths
into a JSON parameters file (saved alongside the model as
`<pseudobulk>_f<fold>.fit_params.json`) and shells out to `cherimoya fit -p
<file>` rather than reimplementing its optimizer/scheduler setup -- that CLI
also runs its own evaluate step on `validation_chroms` afterward, writing
`<pseudobulk>_f<fold>.evaluate.json`/`.performance.tsv` as a side effect
(not a substitute for `benchmark.py`'s held-out test-chromosome
evaluation).

Use `--help` on `negatives.py`/`launch.py` for job selection flags, including
`-p`/`--pseudobulk`, `-f`/`--fold`, `--dataset`, and `--rerun`.
