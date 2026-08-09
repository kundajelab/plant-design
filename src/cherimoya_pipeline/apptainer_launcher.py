"""Shared Apptainer helpers for Cherimoya SLURM launchers.

Cherimoya requires torch>=2.9/triton>=3.5.1; on HPC systems with an old
system toolchain (see pyproject.toml's pybigtools==0.2.5 pin, needed because
newer releases require a source build that fails on Sherlock's assembler),
`uv sync` can be unable to build or resolve these directly. The Cherimoya
Apptainer image (see apptainer/) sidesteps that by bootstrapping from a
prebuilt PyTorch container image instead of the host toolchain. Launchers
that need Cherimoya at runtime (fit/, benchmark/) can opt into it with
--apptainer; the default everywhere else remains `uv run --frozen` inside
the shared plant-design env.
"""

from __future__ import annotations

import os


def default_apptainer_image() -> str:
    return os.environ.get(
        "CHERIMOYA_APPTAINER_IMAGE",
        f"/scratch/users/{os.environ.get('USER', '')}/apptainer/cherimoya.sif",
    )


def add_apptainer_args(parser) -> None:
    parser.add_argument(
        "--apptainer",
        action="store_true",
        help=(
            "Run jobs via `apptainer exec --nv` against the Cherimoya "
            "Apptainer image (see src/cherimoya_pipeline/apptainer/) "
            "instead of `uv run --frozen` inside the shared plant-design "
            "env. Useful on HPC systems (e.g. Sherlock) whose system "
            "toolchain can't build/resolve cherimoya's torch>=2.9/"
            "triton>=3.5.1 requirement."
        ),
    )
    parser.add_argument(
        "--apptainer-image",
        type=str,
        default=default_apptainer_image(),
        help=(
            "Path to the Cherimoya Apptainer .sif image (default: "
            "$CHERIMOYA_APPTAINER_IMAGE, or "
            "/scratch/users/$USER/apptainer/cherimoya.sif)."
        ),
    )
    parser.add_argument(
        "--apptainer-bind",
        action="append",
        default=None,
        help=(
            "Path to bind into the Apptainer container; may be repeated "
            "(default: the repo root -- add more if models/data live "
            "outside it, e.g. on scratch or oak)."
        ),
    )


def resolve_run_prefix(args, repo_root) -> tuple[str, ...]:
    """Return the command prefix to run an executable for this launcher.

    Callers append the executable name themselves (e.g. "python" or
    "cherimoya"), since --apptainer and the default `uv run --frozen` path
    both take that as their next argument.
    """
    if not getattr(args, "apptainer", False):
        return ("uv", "run", "--frozen")
    binds = args.apptainer_bind or [str(repo_root)]
    prefix = ["apptainer", "exec", "--nv"]
    for path in binds:
        prefix.extend(["--bind", str(path)])
    prefix.append(args.apptainer_image)
    return tuple(prefix)
