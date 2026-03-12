#chrombpnet prep nonpeaks [-h] -g GENOME -o OUTPUT_PREFIX -p PEAKS -c CHROM_SIZES -fl CHR_FOLD_PATH [-il INPUTLEN] [-st STRIDE]
#                                [-npr NEG_TO_POS_RATIO_TRAIN] [-br BLACKLIST_REGIONS]

mkdir -p ../../../data/nonpeaks/
pseudobulk="guard"
for i in {0..4}; do
chrombpnet prep nonpeaks \
    -g ../../../data/Sbicolor_730_v5.0.fa \
    -o ../../../data/nonpeaks/${pseudobulk}_fold_${i} \
    -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
    -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
    -fl ../../../configs/fold_${i}.json;
done