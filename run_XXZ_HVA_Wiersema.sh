#!/bin/bash -l
#SBATCH -J XXZ
#SBATCH --qos=normal
#SBATCH -o %j_XXZ.out
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 6-24:00:00

echo "SLURM JOB ID: $SLURM_JOB_ID"
phys=QAOA_XXZ_Wiersema
repeat=3
num_q=14 # 只能是从4-14的偶数,奇数的话报错
layer=$(( 2 * num_q )) 
num_iter_oicd=1000 # 由于XXZ参数变得多, 这里也要变大
num_iter_rcd=1000
r2=$(( num_q / 2 ))
if (( r2 % 2 == 1 )); then
  r2=$(( r2 - 1 ))
fi
r1=$(( r2 / 2 ))
num_iter_gd=$(( (num_iter_oicd * 2 * r2) / (2 * layer * 4 * (r1 + r2)) + 1 ))
Delta=0.5
lr_gd=0.01
lr_rcd=0.02
x_lim_coff=2
y_lim=0

python run_XXZ_HVA_Wiersema.py \
  --num_q ${num_q} \
  --layer ${layer} \
  --lr_gd ${lr_gd} \
  --lr_rcd ${lr_rcd} \
  --num_iter_oicd ${num_iter_oicd} \
  --num_iter_rcd ${num_iter_rcd} \
  --num_iter_gd ${num_iter_gd} \
  --repeat ${repeat} \
  --Delta ${Delta} 

python lai_plot_script.py \
  --phys ${phys} \
  --num_q ${num_q} \
  --lr_gd ${lr_gd} \
  --lr_rcd ${lr_rcd} \
  --layer ${layer} \
  --repeat ${repeat} \
  --x_lim_coff ${x_lim_coff} \
  --y_lim ${y_lim} \
  --Delta ${Delta}
