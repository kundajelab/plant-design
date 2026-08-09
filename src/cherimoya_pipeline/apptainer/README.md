# Building the Cherimoya Apptainer Image

Cherimoya requires `torch>=2.9`/`triton>=3.5.1` (see the root `pyproject.toml`).
On HPC systems with an old system toolchain -- for example Sherlock, whose
assembler forces `pybigtools` to be pinned to `0.2.5` in `pyproject.toml` --
`uv sync` may be unable to build or resolve these directly. This image
sidesteps that by bootstrapping from a prebuilt PyTorch Docker image
(`pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime`, CUDA 12.6, confirmed to
work via the driver's forward-compatibility mechanism on Sherlock in the
sibling `procap-atlas` project) rather than compiling anything against the
host toolchain.

Apptainer is not required to use this pipeline: the default for every
launcher remains `uv run --frozen` inside the shared `plant-design`
environment (see the root `CLAUDE.md`/`README.md`). Pass `--apptainer` to
`src/cherimoya_pipeline/fit/launch.py`, `fit/negatives.py`, or
`benchmark/launch.py` to run their jobs through this image instead, on
clusters where the native install path doesn't work.

## Build

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

The launchers default their image path to `$CHERIMOYA_APPTAINER_IMAGE`, or
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
