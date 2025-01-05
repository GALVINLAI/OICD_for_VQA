#!/bin/bash

phys=tfim_HVA_Wiersema

repeat=10
num_iter=500

num_q=6
layer=8
Delta=0.5

lr_gd=0.01
lr_rcd=0.02

x_lim_coff=1.5
y_lim=0

# python run_tfim_HVA_Wiersema.py --num_q ${num_q} --layer ${layer} --lr_gd ${lr_gd} --lr_rcd ${lr_rcd} --num_iter ${num_iter} --repeat ${repeat} --Delta ${Delta} 
python lai_plot_script.py --phys ${phys} --num_q ${num_q} --layer ${layer} --repeat ${repeat} --x_lim_coff ${x_lim_coff} --y_lim ${y_lim} --Delta ${Delta}