for pb in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem";
    do echo time python benchmark_0.8.py -p ${pb} --finetuned -v; done | \
simple_gpu_scheduler --gpus \
    MIG-166d7783-762d-5f61-b31c-549eb4e0fba0 \
    MIG-f80e9374-504a-571b-bac0-6fb00750db4c \
    MIG-40f43250-998e-586a-ac37-d6520e92590f