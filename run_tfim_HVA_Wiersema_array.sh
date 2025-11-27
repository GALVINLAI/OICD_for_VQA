#!/bin/bash -l

#SBATCH -J TFIM                            # Job name: TFIM
#SBATCH --qos=normal                       # Set quality of service (queue priority level)
#SBATCH -o %j_TFIM.out                     # Output file, %j = job ID (%A = main job ID, %a = array index)
#SBATCH -p cpu                             # Use the 'cpu' partition
#SBATCH -N 1                               # Request 1 node
#SBATCH --cpus-per-task=16                # Request 16 CPUs per task
#SBATCH -t 6-24:00:00                      # Set time limit: 6 days and 24 hours
#SBATCH --array=0                          # Job array index, here only task 3 is submitted
                                           # To submit all 6 tasks (index 0–5), use: --array=0-5

echo "SLURM JOB ID: $SLURM_JOB_ID, ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

phys=tfim_HVA_Wiersema_new
repeat=10

# Number of qubits: increases with array index
num_q=$(( 4 + 2*SLURM_ARRAY_TASK_ID ))    # Values: 4, 6, 8, 10, 12, 14, 16

# Number of layers in the circuit
layer=$(( 2 * num_q ))                    # Alternatively: layer=$(( num_q / 2 ))

# Learning rates for gradient descent (GD) and randomized coordinate descent (RCD)
lr_gd=0.01
lr_rcd=0.02

# Number of iterations for different optimizers
num_iter_oicd=$(( layer * 2 * 100 ))      # e.g., 400 * 100 = 40000
num_iter_rcd=${num_iter_oicd}             # Same as OICD
num_iter_gd=$(( (2 * num_iter_oicd) / (2 * layer * 2) + 1 ))  # Adjusted GD iterations

JOB_ID="${SLURM_JOB_ID}"                  # Store SLURM job ID in a variable

# Run the Python script with arguments
python run_tfim_HVA_Wiersema_new.py \
  --num_q ${num_q} \
  --layer ${layer} \
  --lr_gd ${lr_gd} \
  --lr_rcd ${lr_rcd} \
  --num_iter_oicd ${num_iter_oicd} \
  --num_iter_rcd ${num_iter_rcd} \
  --num_iter_gd ${num_iter_gd} \
  --repeat ${repeat} \
  --JOB_ID "${JOB_ID}"
