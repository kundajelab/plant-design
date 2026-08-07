# Cherimoya tf-MoDISco

Run these commands from the repository root after attribution outputs have been
created in `attributions/`.

Python dependencies are managed by `uv`, so invoke Python entrypoints with
`uv run python`.

## Inputs

MoDISco jobs require matching attribution and OHE files from
`src/cherimoya_pipeline/attribute/`:

```text
attributions/cherimoya/<pseudobulk>_<head>.npz
attributions/cherimoya/<pseudobulk>_ohe.npz
```

For DA-specific runs, filenames include the DA suffix, for example
`_da_specific_union`.

Reports also require:

```text
data/JASPAR2026_CORE_plants_non-redundant_pfms_meme.txt
```

## Run Motif Discovery

All base all-peak MoDISco jobs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py
```

DA-specific jobs on per-cell-type specific-union peaks:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --time 2-00:00:00
```

Finetuned and adapter DA-specific runs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --finetuned \
    --time 2-00:00:00

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --finetuned \
    --da-fraction 0.8 \
    --time 2-00:00:00

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.5 \
    --time 2-00:00:00

uv run python src/cherimoya_pipeline/modisco/launch.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.8 \
    --time 2-00:00:00
```

Use `--dry-run` to inspect the SLURM scripts before submitting. Useful filters:

```bash
uv run python src/cherimoya_pipeline/modisco/launch.py --head count --dry-run
uv run python src/cherimoya_pipeline/modisco/launch.py --head profile --dry-run
```

## Generate Reports

After `.modisco.h5` outputs exist, launch reports:

```bash
uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --time 4:00:00
```

Finetuned and adapter report jobs:

```bash
uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --finetuned \
    --time 4:00:00

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --finetuned \
    --da-fraction 0.8 \
    --time 4:00:00

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.5 \
    --time 4:00:00

uv run python src/cherimoya_pipeline/modisco/launch_report.py \
    --da-peaks specific_union \
    --adapter \
    --da-fraction 0.8 \
    --time 4:00:00
```

Report jobs skip missing motif files, so it is safe to run the report launcher
after a subset of motif jobs has completed.

