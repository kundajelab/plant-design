#!/usr/bin/env bash
# Fetch the classic TAIR10 Arabidopsis thaliana assembly and write it to
# data/A.thaliana/Athaliana_TAIR10.fa with chromosome names matching the
# MoralesArabidopsis pseudobulk peaks/fragments (chr1..chr5, chrM, chrC).
#
# The 5 nuclear chromosomes come from NCBI's GCA_000001735.1 "TAIR10"
# assembly (2010) -- NOT GCA_000001735.2/GCF_000001735.4 "TAIR10.1" (2023),
# whose re-annotated mitochondrion is 367,808 bp instead of the classic
# 366,924 bp that matches this dataset's chrom sizes. The mitochondrion
# (NC_001284.2) and chloroplast (NC_000932.1) are fetched separately via
# NCBI's efetch, since GCA_000001735.1's packaged genomic.fna only includes
# the 5 nuclear chromosomes.
#
# Usage:
#     bash src/data_preprocessing/download_arabidopsis_genome.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/data/A.thaliana"
OUT_FASTA="${OUT_DIR}/Athaliana_TAIR10.fa"

mkdir -p "${OUT_DIR}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Fetching nuclear chromosomes (GCA_000001735.1 TAIR10)..."
curl -sL --fail \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/735/GCA_000001735.1_TAIR10/GCA_000001735.1_TAIR10_genomic.fna.gz" \
    -o "${tmp_dir}/chroms.fna.gz"
gunzip "${tmp_dir}/chroms.fna.gz"

echo "Fetching mitochondrion (NC_001284.2)..."
curl -sL --fail \
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001284.2&rettype=fasta&retmode=text" \
    -o "${tmp_dir}/chrM.fna"

echo "Fetching chloroplast (NC_000932.1)..."
curl -sL --fail \
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000932.1&rettype=fasta&retmode=text" \
    -o "${tmp_dir}/chrC.fna"

python - "${tmp_dir}/chroms.fna" "${tmp_dir}/chrM.fna" "${tmp_dir}/chrC.fna" "${OUT_FASTA}" <<'PYEOF'
import sys


def read_fasta(path):
    records = []
    name, lines = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name:
                    records.append((name, lines))
                name, lines = line[1:].split()[0], []
            else:
                lines.append(line.rstrip("\n"))
        if name:
            records.append((name, lines))
    return records


chroms_path, chrm_path, chrc_path, out_path = sys.argv[1:5]
name_map = {
    "CP002684.1": "chr1",
    "CP002685.1": "chr2",
    "CP002686.1": "chr3",
    "CP002687.1": "chr4",
    "CP002688.1": "chr5",
}
with open(out_path, "w") as out:
    for name, lines in read_fasta(chroms_path):
        out.write(f">{name_map[name]}\n")
        out.write("\n".join(lines) + "\n")
    for label, path in (("chrM", chrm_path), ("chrC", chrc_path)):
        for _, lines in read_fasta(path):
            out.write(f">{label}\n")
            out.write("\n".join(lines) + "\n")
PYEOF

echo "Indexing..."
samtools faidx "${OUT_FASTA}"
cut -f1,2 "${OUT_FASTA}.fai" > "${OUT_DIR}/Athaliana_TAIR10.chrom.sizes"

echo "Wrote ${OUT_FASTA}:"
cat "${OUT_DIR}/Athaliana_TAIR10.chrom.sizes"
