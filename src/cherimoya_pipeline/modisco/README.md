# Cherimoya tf-MoDISco

Run these commands from the repository root after attribution outputs have been
created in `attributions/`.

Python dependencies are managed by `uv`, so invoke Python entrypoints with
`uv run python`.

## Inputs

MoDISco jobs require matching attribution and OHE files from
`src/cherimoya_pipeline/attribute/`:

```text
attributions/<species>/cherimoya/<pseudobulk>_<head>.npz
attributions/<species>/cherimoya/<pseudobulk>_ohe.npz
```

For DA-specific runs, filenames include the DA suffix, for example
`_da_specific_union`.

Reports also require:

```text
data/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt
```

## Run Motif Discovery

`launch.py` submits one SLURM job per (pseudobulk, head) pair whose output
is missing. `modisco` (tf-modisco-lite) is pip-installed directly inside the
Cherimoya Apptainer image (see `../apptainer/`), so jobs run `modisco
motifs` via `apptainer exec` against that image by default (CPU-only, no
`--nv`/GPU needed) -- there's no separate Lmod/conda environment for
modisco. `--dry-run` prints the generated sbatch scripts instead of
submitting; `--local` runs in the foreground instead of submitting to
SLURM.

All base all-peak MoDISco jobs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py
uv run python src/cherimoya_pipeline/modisco/launch.py --dry-run
```

DA-specific jobs on per-cell-type specific-union peaks:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union
```

Finetuned and adapter DA-specific runs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --finetuned

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --finetuned \
    --da-fraction 0.8

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.5

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.8
```

Useful filters:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py --head count
uv run python src/cherimoya_pipeline/modisco/launch.py --head profile
```

## Generate Reports

After `.modisco.h5` outputs exist, submit report jobs the same way (also
via the Apptainer image, CPU-only):

```bash
uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union
```

Finetuned and adapter report jobs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --finetuned

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --finetuned \
    --da-fraction 0.8

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.5

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.8
```

Report jobs skip missing motif files, so it is safe to run the report launcher
after a subset of motif jobs has completed.
