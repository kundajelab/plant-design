#!/usr/bin/env bash
set -euo pipefail

# Create nonpeaks for each pseudobulk and fold

dataset="${DATASET:-base}"
case "${dataset}" in
    base)
        pseudobulks=("bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem")
        peaks_dir="../../../data/peaks"
        nonpeaks_dir="../../../data/nonpeaks"
        ;;
    fine)
        pseudobulks=("epidermis_I" "epidermis_II" "mesophyll_etiolated" "mesophyll_photosynthetic" "meristem_epidermis" "bundle_sheath" "silica_cell" "sclerenchyma" "meristem" "phloem_companion_cell" "xylem" "phloem_parenchyma" "meristem_bundle_sheath" "subsidiary_cell" "guard_cell" "trichome" "ligule")
        peaks_dir="../../../data/peaks_fine"
        nonpeaks_dir="../../../data/nonpeaks_fine"
        ;;
    coarse)
        pseudobulks=("bundle_sheath" "Epidermis_I" "Epidermis_II" "meristem" "mesophyll" "phloem" "sclerenchyma" "xylem")
        peaks_dir="../../../data/peaks_coarse"
        nonpeaks_dir="../../../data/nonpeaks_coarse"
        ;;
    *)
        echo "Unknown DATASET='${dataset}' (expected base, fine, or coarse)" >&2
        exit 1
        ;;
esac

mkdir -p "${nonpeaks_dir}"
for pseudobulk in "${pseudobulks[@]}"; do
    for i in {0..4}; do
        chrombpnet prep nonpeaks \
            -g ../../../data/Sbicolor_730_v5.0.fa \
            -o "${nonpeaks_dir}/${pseudobulk}_fold_${i}" \
            -p "${peaks_dir}/${pseudobulk}_peaks.narrowPeak" \
            -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
            -fl ../../../configs/fold_${i}.json
    done
done
