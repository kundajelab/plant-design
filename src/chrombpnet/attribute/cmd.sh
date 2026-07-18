#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-base}"
case "${dataset}" in
    base)
        pseudobulks=("bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem")
        peaks_dir="../../../data/peaks"
        ;;
    fine)
        pseudobulks=("epidermis_I" "epidermis_II" "mesophyll_etiolated" "mesophyll_photosynthetic" "meristem_epidermis" "bundle_sheath" "silica_cell" "sclerenchyma" "meristem" "phloem_companion_cell" "xylem" "phloem_parenchyma" "meristem_bundle_sheath" "subsidiary_cell" "guard_cell" "trichome" "ligule")
        peaks_dir="../../../data/peaks_fine"
        ;;
    coarse)
        pseudobulks=("bundle_sheath" "Epidermis_I" "Epidermis_II" "meristem" "mesophyll" "phloem" "sclerenchyma" "xylem")
        peaks_dir="../../../data/peaks_coarse"
        ;;
    *)
        echo "Unknown DATASET='${dataset}' (expected base, fine, or coarse)" >&2
        exit 1
        ;;
esac

for pseudobulk in "${pseudobulks[@]}"; do
    echo time python filter_nonACGT_regions.py -v \
        -f ../../../data/Sbicolor_730_v5.0.fa \
        -b ${peaks_dir}/${pseudobulk}_peaks.narrowPeak \
        -o ${peaks_dir}/${pseudobulk}_peaks_ACGT.narrowPeak
done | parallel

for pseudobulk in "${pseudobulks[@]}"; do
    echo time python save_ohe.py -v --dataset ${dataset} -p ${pseudobulk}
done | parallel

if [[ "${dataset}" != "base" ]]; then
    time python launch.py --dataset ${dataset} --model accessibility --time 6:00:00
    exit 0
fi

python ../../data_preprocessing/write_pairwise_higher_beds.py

for pseudobulk in "${pseudobulks[@]}"; do
    echo time python save_ohe.py -v -p ${pseudobulk} --da-peaks specific_union
done | parallel

time python launch.py --da-peaks specific_union --model accessibility --time 6:00:00

time python launch.py --da-peaks specific_union --model accessibility --finetuned --time 6:00:00

time python launch.py --da-peaks specific_union --model accessibility --finetuned --da-fraction 0.8 --time 6:00:00

time python launch.py --da-peaks specific_union --model accessibility --adapter --da-fraction 0.5 --time 6:00:00

time python launch.py --da-peaks specific_union --model accessibility --adapter --da-fraction 0.8 --time 6:00:00
