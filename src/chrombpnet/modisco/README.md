# ChromBPNet tf-MoDISco

Run these commands from the repository root after attribution outputs have been
created in `attributions/`.

Python dependencies are managed by `uv`, so invoke Python entrypoints with
`uv run python`.

## Inputs

MoDISco jobs require matching attribution and OHE files from
`src/chrombpnet/attribute/`:

```text
attributions/chrombpnet/<pseudobulk>_<model_type>_<head>.npz
attributions/chrombpnet/<pseudobulk>_ohe.npz
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
uv run python src/chrombpnet/modisco/launch.py --model accessibility
```

DA-specific jobs on per-cell-type specific-union peaks:

```bash
uv run python src/chrombpnet/modisco/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --time 2-00:00:00
```

Finetuned and adapter DA-specific runs:

```bash
uv run python src/chrombpnet/modisco/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --time 2-00:00:00

uv run python src/chrombpnet/modisco/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --da-fraction 0.8 \
    --time 2-00:00:00

uv run python src/chrombpnet/modisco/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.5 \
    --time 2-00:00:00

uv run python src/chrombpnet/modisco/launch.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.8 \
    --time 2-00:00:00
```

Use `--dry-run` to inspect the SLURM scripts before submitting. Useful filters:

```bash
uv run python src/chrombpnet/modisco/launch.py --model bias --head count --dry-run
uv run python src/chrombpnet/modisco/launch.py --model accessibility --head profile --dry-run
```

## Generate Reports

After `.modisco.h5` outputs exist, launch reports:

```bash
uv run python src/chrombpnet/modisco/launch_report.py \
    --da-peaks specific_union \
    --model accessibility \
    --time 4:00:00
```

Finetuned and adapter report jobs:

```bash
uv run python src/chrombpnet/modisco/launch_report.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --time 4:00:00

uv run python src/chrombpnet/modisco/launch_report.py \
    --da-peaks specific_union \
    --model accessibility \
    --finetuned \
    --da-fraction 0.8 \
    --time 4:00:00

uv run python src/chrombpnet/modisco/launch_report.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.5 \
    --time 4:00:00

uv run python src/chrombpnet/modisco/launch_report.py \
    --da-peaks specific_union \
    --model accessibility \
    --adapter \
    --da-fraction 0.8 \
    --time 4:00:00
```

Report jobs skip missing motif files, so it is safe to run the report launcher
after a subset of motif jobs has completed.

