for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo time python filter_nonACGT_regions.py -v \
        -f ../../../data/Sbicolor_730_v5.0.fa \
        -b ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
        -o ../../../data/peaks/${pseudobulk}_peaks_ACGT.narrowPeak
done | parallel

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo time python save_ohe.py -v -p ${pseudobulk}
done | parallel

time python launch.py

time python launch.py --finetuned --da-fraction 0.8
time python launch.py --da-peaks tau --model accessibility --time 6:00:00
time python launch.py --da-peaks pairwise --model accessibility --time 6:00:00

time python launch.py --da-peaks pairwise --finetuned --da-fraction 0.8 --time 6:00:00
time python launch.py --da-peaks tau --finetuned --da-fraction 0.8 --time 6:00:00
time python launch.py --da-peaks pairwise --finetuned --da-fraction 0.8 --time 6:00:00