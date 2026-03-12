data_dir=/oak/stanford/projects/sorghum/SwiftMultiomeReanalysis/pseudobulk_fragments/
ref_dir=/oak/stanford/projects/sorghum/genomes/SbicolorJGIv5.1/
mkdir -p $SCRATCH/SwiftMultiomeReanalysis/pseudobulk_bigwigs/

rsync --progress $ref_dir/Sbicolor_730_v5.0.fa $SCRATCH/SwiftMultiomeReanalysis/
samtools faidx $SCRATCH/SwiftMultiomeReanalysis/Sbicolor_730_v5.0.fa
cut -f1,2 $SCRATCH/SwiftMultiomeReanalysis//Sbicolor_730_v5.0.fa.fai > $SCRATCH/SwiftMultiomeReanalysis/Sbicolor_730_v5.0.chrom.sizes

for f in "bundle_sheath" "epidermis" "guard" "mesophyll" "phloem" "xylem"
do
    cmd="time bam2bw $data_dir/${f}_fragments.tsv.gz -ps 4 -ns -4 -u -v -s $SCRATCH/SwiftMultiomeReanalysis/Sbicolor_730_v5.0.chrom.sizes -n $SCRATCH/SwiftMultiomeReanalysis/pseudobulk_bigwigs/${f}"
    echo $cmd
    eval $cmd
done

# /oak/stanford/projects/sorghum/SwiftMultiomeReanalysis/pseudobulk_fragments