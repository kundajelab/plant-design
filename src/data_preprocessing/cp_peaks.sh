#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

species="${SPECIES:-S.bicolor}"
dataset="${DATASET:-coarse}"
if [[ "${species}" == "S.bicolor" ]]; then
    data_root="${REPO_ROOT}/data"
else
    data_root="${REPO_ROOT}/data/${species}"
fi

if [[ "${dataset}" == "base" ]]; then
    peaks_dirname="peaks"
else
    peaks_dirname="peaks_${dataset}"
fi

data_dir="${DATA_DIR:-/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_encode_coarse}"
out_dir="${OUT_DIR:-${data_root}/${peaks_dirname}}"
mkdir -p "${out_dir}"

rsync --progress "${data_dir}"/*.narrowPeak "${out_dir}/"
