#!/bin/bash -l
#SBATCH -J nbconvert
#SBATCH --qos=normal
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=18
#SBATCH -t 6-24:00:00

JOBID=${SLURM_JOB_ID}
OUT_NOTEBOOK="executed_${JOBID}.ipynb"

# 执行 Notebook，并输出执行后的 Notebook 或 HTML
jupyter nbconvert \
  --to notebook \
  --execute tfim_HVA_Wiersema_BP.ipynb \
  --output "${OUT_NOTEBOOK}"

