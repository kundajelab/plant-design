export NUMBA_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniforge3/envs/chrombpnet/lib
mamba activate chrombpnet

i=0
for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
echo time python cli.py bias pipeline \
    -bw ../../../data/bigwigs/${pseudobulk}.bw \
    -d "ATAC" \
    -g ../../../data/Sbicolor_730_v5.0.fa \
    -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
    -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
    -n ../../../data/nonpeaks/${pseudobulk}_fold_${i}_negatives.bed \
    -fl ../../../configs/fold_${i}.json \
    -b 0.5 \
    -o ../../../models/${pseudobulk}/bias_f${i}/ \
    -fp f${i};
done | simple_gpu_scheduler --gpus 0 1 2 3

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for i in {1..4}; do
    echo time python cli.py bias train \
        -bw ../../../data/bigwigs/${pseudobulk}.bw \
        -d "ATAC" \
        -g ../../../data/Sbicolor_730_v5.0.fa \
        -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
        -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
        -n ../../../data/nonpeaks/${pseudobulk}_fold_${i}_negatives.bed \
        -fl ../../../configs/fold_${i}.json \
        -b 0.5 \
        -o ../../../models/${pseudobulk}/bias_f${i}/ \
        -fp f${i}
    done
done | simple_gpu_scheduler --gpus 0 1 

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for i in {0..4}; do
    echo time python cli.py train \
        -bw ../../../data/bigwigs/${pseudobulk}.bw \
        -d "ATAC" \
        -g ../../../data/Sbicolor_730_v5.0.fa \
        -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
        -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
        -n ../../../data/nonpeaks/${pseudobulk}_fold_${i}_negatives.bed \
        -fl ../../../configs/fold_${i}.json \
        -b 0.5 \
        -o ../../../models/${pseudobulk}/bias_f${i}/models/f${i}_bias.h5 \
        -fp f${i}
    done
done | simple_gpu_scheduler --gpus 0 1 2
