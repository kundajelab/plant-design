export NUMBA_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniforge3/envs/chrombpnet/lib
mamba activate chrombpnet

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
        -b ../../../models/${pseudobulk}/bias_f${i}/models/f${i}_bias.h5 \
        -o ../../../models/${pseudobulk}/acc_f${i}/
    done
done | simple_gpu_scheduler --gpus 0 1 2 3

cd /srv/scratch/ayhe/plant-design/src/chrombpnet/fit/
pseudobulk="xylem"
i=4
time python cli.py train \
    -bw ../../../data/bigwigs/${pseudobulk}.bw \
    -d "ATAC" \
    -g ../../../data/Sbicolor_730_v5.0.fa \
    -c ../../../data/Sbicolor_730_v5.0.chrom.sizes \
    -p ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
    -n ../../../data/nonpeaks/${pseudobulk}_fold_${i}_negatives.bed \
    -fl ../../../configs/fold_${i}.json \
    -b ../../../models/${pseudobulk}/bias_f${i}/models/f${i}_bias.h5 \
    -o ../../../models/${pseudobulk}/acc_f${i}/
