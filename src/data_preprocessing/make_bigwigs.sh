#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

species="${SPECIES:-S.bicolor}"
dataset="${DATASET:-base}"
data_root="${REPO_ROOT}/data/${species}"
config_dir="${REPO_ROOT}/configs/${species}"

# base/fine/coarse pull fragments from different Oak sources with different
# pseudobulk namespaces (see configs/<species>/pseudobulks{_fine,_coarse}.txt),
# so both the fetch location and the local/output namespacing vary by dataset.
case "${dataset}" in
    base)
        default_data_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_fragments"
        default_ref_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data"
        dataset_suffix=""
        ;;
    fine)
        default_data_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_encode_fine"
        default_ref_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data"
        dataset_suffix="_fine"
        ;;
    coarse)
        default_data_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data/pseudobulk_encode_coarse"
        default_ref_dir="/oak/stanford/groups/akundaje/ayhe/plant-design/data"
        dataset_suffix="_coarse"
        ;;
    *)
        echo "Unknown DATASET '${dataset}' (expected base, fine, or coarse)." >&2
        exit 1
        ;;
esac

data_dir="${DATA_DIR:-${default_data_dir}}"
ref_dir="${REF_DIR:-${default_ref_dir}}"
out_dir="${OUT_DIR:-${data_root}/bigwigs${dataset_suffix}}"
# Fragments are always synced into data/<species>/ before bam2bw reads them --
# never read directly out of DATA_DIR -- so runs are reproducible from the
# namespaced data/ tree alone and don't require live Oak access on every run.
fragments_dir="${FRAGMENTS_DIR:-${data_root}/fragments${dataset_suffix}}"
# Resolved from configs/<species>/species.json (the same manifest
# species_config.genome_fasta() reads) unless overridden. Must NOT default
# to Sorghum's filename here -- SPECIES=A.thaliana without this would
# silently look for data/A.thaliana/Sbicolor_730_v5.0.fa instead of
# Athaliana_TAIR10.fa.
species_json="${config_dir}/species.json"
genome_name="${GENOME_FASTA_NAME:-$(python -c "
import json, sys
print(json.load(open(sys.argv[1]))['genome_fasta'])
" "${species_json}")}"
genome_fasta="${data_root}/${genome_name}"
chrom_sizes="${genome_fasta%.fa}.chrom.sizes"
pseudobulks_file="${PSEUDOBULKS_FILE:-${config_dir}/pseudobulks${dataset_suffix}.txt}"
# Sorghum's fragment files are named <pseudobulk>_fragments.tsv.gz; other
# sources (e.g. the Arabidopsis MoralesArabidopsis dataset) use
# <pseudobulk>.fragments.tsv.gz -- override via FRAGMENTS_SUFFIX.
fragments_suffix="${FRAGMENTS_SUFFIX:-_fragments.tsv.gz}"

mkdir -p "${out_dir}" "${data_root}" "${fragments_dir}"

# Skip the oak rsync entirely if the genome is already in place (e.g. fetched
# directly from a public archive instead of an oak ref_dir, as for
# A.thaliana's TAIR10 assembly) -- avoids requiring a valid REF_DIR for
# species that don't have one.
if [[ ! -f "${genome_fasta}" ]]; then
    rsync --ignore-existing --progress "${ref_dir}/${genome_name}" "${genome_fasta}"
fi
if [[ ! -f "${genome_fasta}.fai" ]]; then
    samtools faidx "${genome_fasta}"
fi
if [[ ! -f "${chrom_sizes}" ]]; then
    cut -f1,2 "${genome_fasta}.fai" > "${chrom_sizes}"
fi

while IFS= read -r f || [[ -n "${f}" ]]
do
    [[ -z "${f}" || "${f}" == \#* ]] && continue

    local_fragments="${fragments_dir}/${f}${fragments_suffix}"
    if [[ ! -f "${local_fragments}" ]]; then
        rsync --ignore-existing --progress "${data_dir}/${f}${fragments_suffix}" "${local_fragments}"
    fi

    echo "time uv run bam2bw ${local_fragments} -ps 4 -ns -4 -u -f -v -s ${chrom_sizes} -n ${out_dir}/${f}"
    time uv --directory "${REPO_ROOT}" run bam2bw "${local_fragments}" \
        -ps 4 \
        -ns -4 \
        -u \
        -f \
        -v \
        -s "${chrom_sizes}" \
        -n "${out_dir}/${f}"
done < "${pseudobulks_file}"
