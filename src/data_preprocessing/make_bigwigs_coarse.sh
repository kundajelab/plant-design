#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

data_dir="${DATA_DIR:-/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_encode_coarse}"
ref_dir="${REF_DIR:-/oak/stanford/groups/akundaje/ayhe/plant-design/data}"
out_dir="${OUT_DIR:-${REPO_ROOT}/data/bigwigs_coarse}"
genome_fasta="${REPO_ROOT}/data/Sbicolor_730_v5.0.fa"
chrom_sizes="${REPO_ROOT}/data/Sbicolor_730_v5.0.chrom.sizes"

mkdir -p "${out_dir}" "${REPO_ROOT}/data"

rsync --progress "${ref_dir}/Sbicolor_730_v5.0.fa" "${genome_fasta}"
samtools faidx "${genome_fasta}"
cut -f1,2 "${genome_fasta}.fai" > "${chrom_sizes}"

for f in "bundle_sheath" "Epidermis_I" "Epidermis_II" "meristem" "mesophyll" "phloem" "sclerenchyma" "xylem"
do
    echo "time uv run bam2bw ${data_dir}/${f}_fragments.tsv.gz -ps 4 -ns -4 -u -v -s ${chrom_sizes} -n ${out_dir}/${f}"
    time uv run bam2bw "${data_dir}/${f}_fragments.tsv.gz" \
        -ps 4 \
        -ns -4 \
        -u \
        -v \
        -s "${chrom_sizes}" \
        -n "${out_dir}/${f}"
done
