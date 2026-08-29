mkdir -p ../../../finemo

for p in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    for head in "count" "profile"; do 
        echo finemo extract-regions-modisco-fmt \
            -s ../../../attributions/${p}_ohe.npz \
            -a ../../../attributions/${p}_accessibility_${head}.npz \
            -o ../../../finemo/chrombpnet/${p}_accessibility_${head}_finemo.npz \
            -p ../../../data/peaks/${p}_peaks_ACGT.narrowPeak \
            -w 2114
    done
done

for p in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo finemo call-hits -b 1000 \
        -r ../../../finemo/chrombpnet/${p}_accessibility_count_finemo.npz \
        -m ../../../modisco/chrombpnet/${p}_accessibility_count.modisco.h5 \
        -o ../../../finemo/${p}_accessibility_count_finemo
done | simple_gpu_scheduler --gpus 0 1

for p in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"; do
    echo finemo call-hits -b 1000 \
        -r ../../../finemo/chrombpnet/${p}_accessibility_profile_finemo.npz \
        -m ../../../modisco/chrombpnet/${p}_accessibility_profile.modisco.h5 \
        -o ../../../finemo/${p}_accessibility_profile_finemo
done | simple_gpu_scheduler --gpus 0 1