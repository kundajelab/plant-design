#!/bin/bash -l
#SBATCH --job-name=mpx_deepliftshap
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH -C GPU_SKU:A100_SXM4|GPU_SKU:L40S|GPU_SKU:H100_SXM5|GPU_SKU:RTX_3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu,akundaje,owners
#SBATCH --time=48:00:00
#SBATCH --array=0-3
#SBATCH --output=$REPO_ROOT/logs/%x_%A_%a.out
#SBATCH --error=$REPO_ROOT/logs/%x_%A_%a.err

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")
mkdir -p "$REPO_ROOT/logs"

ml openblas/0.3.28
ml xsimd/8.1.0
ml xz/5.8.1
ml hdf5/1.14.4
ml arrow/22.0.0
ml load py-pyarrow/18.1.0_py312
ml lz4/1.8.0
ml biology
ml htslib
ml ucsc-utils

mamba activate torch
nvidia-smi -L

pseudobulks=("phloem" "phloem" "xylem" "xylem")
heads=("profile" "count" "profile" "count")

pseudobulk=${pseudobulks[$SLURM_ARRAY_TASK_ID]}
head=${heads[$SLURM_ARRAY_TASK_ID]}

scontrol update JobId=$SLURM_JOB_ID JobName=deepliftshap_${pseudobulk}_${head}

time python deepliftshap.py -v -p ${pseudobulk} -m accessibility --head ${head}
