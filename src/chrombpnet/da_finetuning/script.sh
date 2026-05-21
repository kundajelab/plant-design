mamba activate torch

export NUMBA_NUM_THREADS=4

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for f in {0..4}; do echo time python finetune_da.py -p ${pseudobulk} -f $f -v; done
done | simple_gpu_scheduler --gpus \
    MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 \
    MIG-f80e9374-504a-571b-bac0-6fb00750db4c \
    MIG-40f43250-998e-586a-ac37-d6520e92590f


mamba activate torch

export NUMBA_NUM_THREADS=4

for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for f in {0..4}; do echo time python finetune_da.py -p ${pseudobulk} -f $f --da-fraction 0.8 -v; done
done | simple_gpu_scheduler --gpus \
    MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 \
    MIG-f80e9374-504a-571b-bac0-6fb00750db4c \
    MIG-40f43250-998e-586a-ac37-d6520e92590f


mamba activate torch

export NUMBA_NUM_THREADS=4

for da_fraction in 0.5 0.8; do
    for pseudobulk in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
        for f in {0..4}; do echo time python finetune_da.py -p ${pseudobulk} -f $f --mode adapter --da-fraction ${da_fraction} -v; done
    done
done | simple_gpu_scheduler --gpus \
    MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 \
    MIG-f80e9374-504a-571b-bac0-6fb00750db4c \
    MIG-40f43250-998e-586a-ac37-d6520e92590f
