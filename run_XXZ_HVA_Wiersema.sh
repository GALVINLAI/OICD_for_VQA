#!/bin/bash

phys=QAOA_XXZ_Wiersema

repeat=10
num_iter=300

num_q=6 # 4
layer=3 # 2
Delta=0.5

lr_gd=0.01
lr_rcd=0.02

x_lim_coff=2
y_lim=0

python run_XXZ_HVA_Wiersema.py --num_q ${num_q} --layer ${layer} --lr_gd ${lr_gd} --lr_rcd ${lr_rcd} --num_iter ${num_iter} --repeat ${repeat} --Delta ${Delta} 
python lai_plot_script.py --phys ${phys} --num_q ${num_q} --lr_gd ${lr_gd} --lr_rcd ${lr_rcd} --layer ${layer} --repeat ${repeat} --x_lim_coff ${x_lim_coff} --y_lim ${y_lim} --Delta ${Delta}
