# Building the Cherimoya Apptainer Image

Cherimoya requires `torch>=2.9`/`triton>=3.5.1` (see the root `pyproject.toml`).
Sherlock's system toolchain is old enough that building/resolving these
directly via `uv sync` can be unreliable. This image sidesteps that by
bootstrapping from a prebuilt PyTorch Docker image
(`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`, CUDA 12.6) rather than
compiling anything against the host toolchain. CUDA 12.6 is the version the
sibling `procap-atlas` project's Cherimoya pipeline uses and has confirmed
works via the driver's forward-compatibility mechanism on Sherlock, across
every GPU SKU it tested (`A100_PCIE`, `A100_SXM4`, `A40`, `H100_SXM5`,
`H200_SXM5`, `L40S`, `RTX_3090`); `check_gpu.py` (see "Running" below) runs
the same empirical check against this repo's own image before trusting it.

By default, every launcher in `src/cherimoya_pipeline/` (`fit/`, `benchmark/`,
`attribute/`, `modisco/`, `da_finetuning/`) submits SLURM jobs that run
through this image via `apptainer exec --nv`. `--local` runs the same
Apptainer-backed commands in the foreground instead of submitting to SLURM
(useful on a GPU box you already have a shell on, e.g. the akundaje Oak GPU
nodes). Apptainer is not required to use this pipeline outside Sherlock/HPC
contexts -- preprocessing and any ad hoc work still uses `uv run` directly
inside the shared `plant-design` environment.

`modisco motifs`/`modisco report` also run through this image (it pip
installs `modisco`, i.e. tf-modisco-lite, directly) -- there's no separate
Lmod/conda environment for modisco.

## Build

SRCC support notes that building directly on Sherlock can be unreliable
(`apptainer build` there is "a bit spotty"); building on a separate machine
that has Apptainer installed and transferring the resulting `.sif` is the
more reliable path, though building directly on Sherlock with the
workarounds in "Troubleshooting" below is also an option:

```bash
# On a separate machine with Apptainer installed:
apptainer build cherimoya.sif cherimoya.def
scp cherimoya.sif sherlock:/scratch/users/$USER/apptainer/cherimoya.sif

# Or directly on Sherlock (see "Troubleshooting" below if this fails):
apptainer build cherimoya.sif cherimoya.def
```

The definition installs Cherimoya and daicon with `--no-deps` after
installing the non-PyTorch dependencies. This preserves the base image's own
`torch`/`torchvision` build; letting pip resolve Cherimoya's
`torch>=2.9.0` requirement itself can replace it with a different PyPI Torch
build and leave companion packages incompatible.

Unlike a conda-managed base image, `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`
is Ubuntu 24.04-based and uses that distro's system Python, which PEP 668
protects from bare `pip install` (`error: externally-managed-environment`).
Every `pip install` in `cherimoya.def` passes `--break-system-packages` for
this reason -- safe here since the image is single-purpose and there's no
shared system pip for it to conflict with.

## Running

```bash
apptainer exec --nv cherimoya.sif python ...
```

Every launcher defaults its image path to `$CHERIMOYA_APPTAINER_IMAGE`, or
`/scratch/users/$USER/apptainer/cherimoya.sif` if that's unset; override
either the env var or `--apptainer-image` if the built `.sif` lives
somewhere else. `--apptainer-bind` (repeatable, default: the repo root)
controls what's mounted into the container -- add more binds if models or
data live outside the repo checkout (e.g. on scratch or oak).

If CUDA initialization fails inside the container with something like
`RuntimeError: ... Error 803: system has unsupported display driver / cuda
driver combination`, try adding `--writable-tmpfs` to the `apptainer exec`
invocation as a first troubleshooting step (needed by some NVIDIA-authored
container entrypoints that reset `/usr/local/cuda/compat/lib` at startup on
Apptainer's read-only-by-default filesystem; the official `pytorch/pytorch`
images don't carry that entrypoint, so this most likely won't be needed
here).

Sherlock does not expose GPU driver or CUDA version as a queryable SLURM
node feature or constraint (only `GPU_BRD`/`GPU_GEN`/`GPU_SKU`/`GPU_MEM`/
`GPU_CC`), so there is no way to target a compatible node ahead of time.
`check_gpu.py` submits one short, minimal-resource job per GPU SKU that runs
`nvidia-smi` and then actually moves a tensor onto the GPU inside this
image, which is the real compatibility test:

```bash
python src/cherimoya_pipeline/apptainer/check_gpu.py --dry-run
python src/cherimoya_pipeline/apptainer/check_gpu.py
python src/cherimoya_pipeline/apptainer/check_gpu.py --skus GPU_SKU:L40S GPU_SKU:A100_PCIE
```

Check `logs/cherimoya_check_gpu/*.err` once the jobs finish -- a line
reading `OK: moved tensor to cuda:0` means that SKU (or rather, whichever
node it happened to land on) works. Re-run this if Sherlock's driver is ever
updated, or before trusting the image on a new cluster.

## Troubleshooting

The base image is large (CUDA libraries included), so the final
`mksquashfs` packing step needs meaningful memory and scratch space when
building directly on Sherlock:

- `FATAL: ... mksquashfs command failed: exit status 139` is a segfault, not
  a build-script error; on a memory-constrained host (e.g. a 16G dev node)
  it usually means `mksquashfs`'s default multi-threaded xz compression ran
  out of RAM. Cap its memory/processor use before building:

  ```bash
  export APPTAINER_MKSQUASHFS_MEM=2G
  export APPTAINER_MKSQUASHFS_PROCS=1
  apptainer build cherimoya.sif cherimoya.def
  ```

- If the build instead runs out of disk space while staging the rootfs
  (default `/tmp`, often small on login nodes), point the temp/cache dirs at
  scratch space. The path **must be absolute** and must already exist -- a
  relative path fails with `FATAL: Unable to create build: failed to find
  mount point for ./apptainer_tmp: no parent mount point found`, since
  Apptainer's fakeroot build needs to resolve which real filesystem mount
  the tmpdir is on:

  ```bash
  export APPTAINER_TMPDIR="$(pwd)/apptainer_tmp"   # or an absolute scratch path
  export APPTAINER_CACHEDIR="$(pwd)/apptainer_cache"
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  apptainer build cherimoya.sif cherimoya.def
  ```

  The `INFO: User not listed in /etc/subuid, trying root-mapped namespace`
  line that can appear alongside this is a benign fallback notice (no
  subuid/subgid mapping granted on that host), not the cause of a failed
  build.
