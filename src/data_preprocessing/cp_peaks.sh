#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

data_dir="${DATA_DIR:-/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_encode_coarse}"
out_dir="${OUT_DIR:-${REPO_ROOT}/data/peaks_coarse}"
mkdir -p "${out_dir}"

rsync --progress "${data_dir}/*.narrowPeak" "${out_dir}"
