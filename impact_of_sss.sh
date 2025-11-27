#!/bin/bash -l
#SBATCH -J nbconvert
#SBATCH --qos=normal
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 6-24:00:00

JOBID=${SLURM_JOB_ID}
OUT_NOTEBOOK="executed_impact_of_sss_${JOBID}.ipynb"

# 执行 Notebook，并输出执行后的 Notebook 或 HTML
# executed_465696.ipynb
# impact_of_sss.ipynb
jupyter nbconvert \
  --to notebook \
  --execute impact_of_sss.ipynb \
  --output "${OUT_NOTEBOOK}"

