GPUS="MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 MIG-f80e9374-504a-571b-bac0-6fb00750db4c MIG-40f43250-998e-586a-ac37-d6520e92590f"

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do 
    for i in {0..4}; do \
        echo python report.py \
            ../../../models/${pseudobulk}/torch/${pseudobulk}_f${i}.bias.torch \
            ../../../data/Sbicolor_730_v5.0.fa \
            ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
            ../../../data/tn5.meme.txt \
            ../../../marginalization/${pseudobulk}/tn5/bias_${i}/ \
            --verbose --random_state 47
    done
done | simple_gpu_scheduler --gpus $GPUS

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do 
    for i in {0..4}; do \
        echo python report.py \
            ../../../models/${pseudobulk}/torch/${pseudobulk}_f${i}.accessibility.torch \
            ../../../data/Sbicolor_730_v5.0.fa \
            ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
            ../../../data/tn5.meme.txt \
            ../../../marginalization/${pseudobulk}/tn5/accessibility_${i}/ \
            --verbose --random_state 47
    done
done | simple_gpu_scheduler --gpus $GPUS
