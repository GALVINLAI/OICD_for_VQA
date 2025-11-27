#!/bin/bash -l
#SBATCH -J TFIM
#SBATCH --qos=normal
#SBATCH -o %j_TFIM.out
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 6-24:00:00

echo "SLURM JOB ID: $SLURM_JOB_ID"
phys=tfim_HVA_Wiersema_new
repeat=1
num_q=4 # <= 14
# layer=$(( 2 * num_q ))
layer=2
num_iter_oicd=500
num_iter_rcd=500
num_iter_gd=$(( 1000 / (2 * layer * 2) + 1 ))
Delta=0.5
lr_gd=0.01
lr_rcd=0.02
# x_lim_coff=1.5
# y_lim=0

python run_tfim_HVA_Wiersema_new.py \
  --num_q ${num_q} \
  --layer ${layer} \
  --lr_gd ${lr_gd} \
  --lr_rcd ${lr_rcd} \
  --num_iter_oicd ${num_iter_oicd} \
  --num_iter_rcd ${num_iter_rcd} \
  --num_iter_gd ${num_iter_gd} \
  --repeat ${repeat} \
  --Delta ${Delta} \
  --JOB_ID ${SLURM_JOB_ID}

# python lai_plot_script.py \
#   --phys ${phys} \
#   --num_q ${num_q} \
#   --lr_gd ${lr_gd} \
#   --lr_rcd ${lr_rcd} \
#   --layer ${layer} \
#   --repeat ${repeat} \
#   --x_lim_coff ${x_lim_coff} \
#   --y_lim ${y_lim} \
#   --Delta ${Delta}
