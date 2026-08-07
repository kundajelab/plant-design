#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

species="${SPECIES:-S.bicolor}"
if [[ "${species}" == "S.bicolor" ]]; then
    data_root="${REPO_ROOT}/data"
    config_dir="${REPO_ROOT}/configs"
else
    data_root="${REPO_ROOT}/data/${species}"
    config_dir="${REPO_ROOT}/configs/${species}"
fi

data_dir="${DATA_DIR:-/oak/stanford/projects/sorghum/SwiftMultiomeReanalysis/pseudobulk_fragments}"
ref_dir="${REF_DIR:-/oak/stanford/projects/sorghum/genomes/SbicolorJGIv5.1}"
out_dir="${OUT_DIR:-${data_root}/bigwigs}"
genome_name="${GENOME_FASTA_NAME:-Sbicolor_730_v5.0.fa}"
genome_fasta="${data_root}/${genome_name}"
chrom_sizes="${genome_fasta%.fa}.chrom.sizes"
pseudobulks_file="${PSEUDOBULKS_FILE:-${config_dir}/pseudobulks.txt}"

mkdir -p "${out_dir}" "${data_root}"

rsync --progress "${ref_dir}/${genome_name}" "${genome_fasta}"
samtools faidx "${genome_fasta}"
cut -f1,2 "${genome_fasta}.fai" > "${chrom_sizes}"

while IFS= read -r f || [[ -n "${f}" ]]
do
    [[ -z "${f}" || "${f}" == \#* ]] && continue
    echo "time uv run bam2bw ${data_dir}/${f}_fragments.tsv.gz -ps 4 -ns -4 -u -v -s ${chrom_sizes} -n ${out_dir}/${f}"
    time uv --directory "${REPO_ROOT}" run bam2bw "${data_dir}/${f}_fragments.tsv.gz" \
        -ps 4 \
        -ns -4 \
        -u \
        -v \
        -s "${chrom_sizes}" \
        -n "${out_dir}/${f}"
done < "${pseudobulks_file}"
