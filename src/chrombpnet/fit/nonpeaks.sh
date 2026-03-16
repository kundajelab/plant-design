#!/bin/bash
# Create nonpeaks for each pseudobulk and fold

mkdir -p ../../../data/nonpeaks/
for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for i in {0..4}; do
        chrombpnet prep nonpeaks \
            -g ../../../data/Sbicolor_730_v5.0.fa \
            -o ../../../data/nonpeaks/${pseudobulk}_fold_${i} \
            -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
            -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
            -fl ../../../configs/fold_${i}.json
    done
done