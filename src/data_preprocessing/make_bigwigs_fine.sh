#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

data_dir="${DATA_DIR:-/oak/stanford/projects/sorghum/SwiftMultiomeReanalysis/pseudobulk_encode_fine}"
ref_dir="${REF_DIR:-/oak/stanford/projects/sorghum/genomes/SbicolorJGIv5.1}"
out_dir="${OUT_DIR:-${REPO_ROOT}/data/bigwigs_fine}"
genome_fasta="${REPO_ROOT}/data/Sbicolor_730_v5.0.fa"
chrom_sizes="${REPO_ROOT}/data/Sbicolor_730_v5.0.chrom.sizes"

mkdir -p "${out_dir}" "${REPO_ROOT}/data"

rsync --progress "${ref_dir}/Sbicolor_730_v5.0.fa" "${genome_fasta}"
samtools faidx "${genome_fasta}"
cut -f1,2 "${genome_fasta}.fai" > "${chrom_sizes}"

for f in "epidermis_I" "epidermis_II" "mesophyll_etiolated" "mesophyll_photosynthetic" "meristem_epidermis" "bundle_sheath" "silica_cell" "sclerenchyma" "meristem" "phloem_companion_cell" "xylem" "phloem_parenchyma" "meristem_bundle_sheath" "subsidiary_cell" "guard_cell" "trichome" "ligule"
do
    echo "time uv run bam2bw ${data_dir}/${f}_fragments.tsv.gz -ps 4 -ns -4 -u -v -s ${chrom_sizes} -n ${out_dir}/${f}"
    time uv --directory "${REPO_ROOT}" run bam2bw "${data_dir}/${f}_fragments.tsv.gz" \
        -ps 4 \
        -ns -4 \
        -u \
        -v \
        -s "${chrom_sizes}" \
        -n "${out_dir}/${f}"
done
