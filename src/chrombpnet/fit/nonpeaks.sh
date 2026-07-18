#!/usr/bin/env bash
set -euo pipefail

# Print one nonpeak-generation command per pseudobulk and fold.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

dataset="${DATASET:-base}"
case "${dataset}" in
    base)
        pseudobulks_file="${repo_root}/configs/pseudobulks.txt"
        peaks_dir="${repo_root}/data/peaks"
        nonpeaks_dir="${repo_root}/data/nonpeaks"
        ;;
    fine)
        pseudobulks_file="${repo_root}/configs/pseudobulks_fine.txt"
        peaks_dir="${repo_root}/data/peaks_fine"
        nonpeaks_dir="${repo_root}/data/nonpeaks_fine"
        ;;
    coarse)
        pseudobulks_file="${repo_root}/configs/pseudobulks_coarse.txt"
        peaks_dir="${repo_root}/data/peaks_coarse"
        nonpeaks_dir="${repo_root}/data/nonpeaks_coarse"
        ;;
    *)
        echo "Unknown DATASET='${dataset}' (expected base, fine, or coarse)" >&2
        exit 1
        ;;
esac

while IFS= read -r pseudobulk || [[ -n "${pseudobulk}" ]]; do
    [[ -z "${pseudobulk}" || "${pseudobulk}" == \#* ]] && continue
    for i in {0..4}; do
        printf "mkdir -p %q && " "${nonpeaks_dir}"
        printf "%q " \
            chrombpnet prep nonpeaks \
            -g "${repo_root}/data/Sbicolor_730_v5.0.fa" \
            -o "${nonpeaks_dir}/${pseudobulk}_fold_${i}" \
            -p "${peaks_dir}/${pseudobulk}_peaks.narrowPeak" \
            -c "${repo_root}/data/Sbicolor_730_v5.0.chrom.sizes" \
            -fl "${repo_root}/configs/fold_${i}.json"
        printf "\n"
    done
done < "${pseudobulks_file}"
