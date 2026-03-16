for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo time python filter_nonACGT_regions.py -v \
        -f ../../../data/Sbicolor_730_v5.0.fa \
        -b ../../../data/peaks/${pseudobulk}_peaks.narrowPeak \
        -o ../../../data/peaks/${pseudobulk}_peaks_ACGT.narrowPeak
done | parallel

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo time python save_ohe.py -v -p ${pseudobulk}
done | parallel

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem"; do
    for head in "profile" "count"; do
        echo time python deepliftshap.py -v -p ${pseudobulk} -m bias --head ${head} -b 256
    done
done | simple_gpu_scheduler --gpus 0 1 2 3

for pseudobulk in "mesophyll" "phloem" "xylem"; do
    for head in "profile" "count"; do
        echo time python deepliftshap.py -v -p ${pseudobulk} -m accessibility --head ${head} -b 32
    done
done | \
    simple_gpu_scheduler --gpus MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 MIG-f80e9374-504a-571b-bac0-6fb00750db4c MIG-40f43250-998e-586a-ac37-d6520e92590f


pseudobulk="xylem"
for head in "profile" "count"; do
    time python deepliftshap.py -v -p ${pseudobulk} -m bias --head ${head} -b 256
done