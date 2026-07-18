#!/usr/bin/env bash
set -euo pipefail

export NUMBA_NUM_THREADS=4
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:~/miniforge3/envs/chrombpnet/lib"
mamba activate chrombpnet

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

dataset="${DATASET:-base}"
case "${dataset}" in
    base)
        pseudobulks=("bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem")
        bigwig_dir="${repo_root}/data/bigwigs"
        peaks_dir="${repo_root}/data/peaks"
        nonpeaks_dir="${repo_root}/data/nonpeaks"
        model_root="${repo_root}/models"
        ;;
    fine)
        pseudobulks=("epidermis_I" "epidermis_II" "mesophyll_etiolated" "mesophyll_photosynthetic" "meristem_epidermis" "bundle_sheath" "silica_cell" "sclerenchyma" "meristem" "phloem_companion_cell" "xylem" "phloem_parenchyma" "meristem_bundle_sheath" "subsidiary_cell" "guard_cell" "trichome" "ligule")
        bigwig_dir="${repo_root}/data/bigwigs_fine"
        peaks_dir="${repo_root}/data/peaks_fine"
        nonpeaks_dir="${repo_root}/data/nonpeaks_fine"
        model_root="${repo_root}/models/fine"
        ;;
    coarse)
        pseudobulks=("bundle_sheath" "Epidermis_I" "Epidermis_II" "meristem" "mesophyll" "phloem" "sclerenchyma" "xylem")
        bigwig_dir="${repo_root}/data/bigwigs_coarse"
        peaks_dir="${repo_root}/data/peaks_coarse"
        nonpeaks_dir="${repo_root}/data/nonpeaks_coarse"
        model_root="${repo_root}/models/coarse"
        ;;
    *)
        echo "Unknown DATASET='${dataset}' (expected base, fine, or coarse)" >&2
        exit 1
        ;;
esac

i=0
for pseudobulk in "${pseudobulks[@]}"; do
echo time python cli.py bias pipeline \
    -bw ${bigwig_dir}/${pseudobulk}.bw \
    -d "ATAC" \
    -g ${repo_root}/data/Sbicolor_730_v5.0.fa \
    -c ${repo_root}/data/Sbicolor_730_v5.0.chrom.sizes \
    -p ${peaks_dir}/${pseudobulk}_peaks.narrowPeak \
    -n ${nonpeaks_dir}/${pseudobulk}_fold_${i}_negatives.bed \
    -fl ${repo_root}/configs/fold_${i}.json \
    -b 0.5 \
    -o ${model_root}/${pseudobulk}/bias_f${i}/ \
    -fp f${i};
done | simple_gpu_scheduler --gpus 0 1 2 3

for pseudobulk in "${pseudobulks[@]}"; do
    for i in {1..4}; do
    echo time python cli.py bias train \
        -bw ${bigwig_dir}/${pseudobulk}.bw \
        -d "ATAC" \
        -g ${repo_root}/data/Sbicolor_730_v5.0.fa \
        -c ${repo_root}/data/Sbicolor_730_v5.0.chrom.sizes \
        -p ${peaks_dir}/${pseudobulk}_peaks.narrowPeak \
        -n ${nonpeaks_dir}/${pseudobulk}_fold_${i}_negatives.bed \
        -fl ${repo_root}/configs/fold_${i}.json \
        -b 0.5 \
        -o ${model_root}/${pseudobulk}/bias_f${i}/ \
        -fp f${i}
    done
done | simple_gpu_scheduler --gpus 0 1 

for pseudobulk in "${pseudobulks[@]}"; do
    for i in {0..4}; do
    echo time python cli.py train \
        -bw ${bigwig_dir}/${pseudobulk}.bw \
        -d "ATAC" \
        -g ${repo_root}/data/Sbicolor_730_v5.0.fa \
        -c ${repo_root}/data/Sbicolor_730_v5.0.chrom.sizes \
        -p ${peaks_dir}/${pseudobulk}_peaks.narrowPeak \
        -n ${nonpeaks_dir}/${pseudobulk}_fold_${i}_negatives.bed \
        -fl ${repo_root}/configs/fold_${i}.json \
        -b 0.5 \
        -o ${model_root}/${pseudobulk}/bias_f${i}/models/f${i}_bias.h5 \
        -fp f${i}
    done
done | simple_gpu_scheduler --gpus 0 1 2
