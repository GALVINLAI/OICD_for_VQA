import argparse
import os, shutil, pickle
import numpy as np
import matplotlib.pyplot as plt

from algo.oicd import oicd
from algo.gd import gd
from algo.rcd import rcd

from qiskit_algorithms import NumPyMinimumEigensolver
from plot_results import plot_results

from algo.utils import (
    xxz_hamiltonian,
    circuit_QAOA_XXZ,
    expectation_loss_grad,
    interp_matrix,
    mse, load,
    optimal_interp_points
)

def create_parser():
    parser = argparse.ArgumentParser(description="A simple argument parser")

    # Problem settings
    parser.add_argument('--num_q', type=int, default=8, help='Number of qubits')
    parser.add_argument('--Delta', type=float, default=1.5, help='XXZ Hamiltonian parameter')
    parser.add_argument('--layer', type=int, default=16, help='Number of circuit layers')
    parser.add_argument('--n_shot', type=int, default=1024, help='Shots per evaluation')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat the experiment')
    parser.add_argument('--tol', type=float, default=1e-2, help='Stopping tolerance')
   
    # Learning rates
    parser.add_argument('--lr_gd', type=float, default=0.01, help='Learning rate for Gradient Descent')
    parser.add_argument('--lr_rcd', type=float, default=0.02, help='Learning rate for Random Coordinate Descent')

    # Iteration counts for optimizers
    parser.add_argument('--num_iter_oicd', type=int, default=500, help='Iterations for OICD optimization')
    parser.add_argument('--num_iter_rcd', type=int, default=500, help='Iterations for RCD optimization')
    parser.add_argument('--num_iter_gd', type=int, default=1, help='Iterations for GD optimization')

    parser.add_argument('--JOB_ID', type=int, default=0, help='SLURM JOB ID')
    parser.add_argument('--phys', type=str, default="XXZ_HVA_Wiersema_new", help='Model name')

    return parser

def print_config(args):
    print("Running various algorithms for the XXZ_HVA_Wiersema model")
    print(f"Job ID                         : {args.JOB_ID}")
    print(f"Number of qubits               : {args.num_q}")
    print(f"Number of circuit layers       : {args.layer}")
    print(f"Number of parameters           : {args.layer * 4}")
    print(f"Number of shots per evaluation : {args.n_shot}")
    print(f"Repeat count                   : {args.repeat}")
    print(f"Hamiltonian parameter Δ        : {args.Delta}")
    print(f"Learning rate [GD]             : {args.lr_gd}")
    print(f"Learning rate [RCD]            : {args.lr_rcd}")
    print(f"Iterations [GD]                : {args.num_iter_gd}")
    print(f"Iterations [RCD]               : {args.num_iter_rcd}")
    print(f"Iterations [OICD]              : {args.num_iter_oicd}")
    print(f"Tolerance                      : {args.tol}")

args = create_parser().parse_args()
print_config(args)

# Extract arguments to variables if needed
repeat = args.repeat
lr_gd = args.lr_gd
lr_rcd = args.lr_rcd
num_iter_oicd = args.num_iter_oicd
num_iter_rcd = args.num_iter_rcd
num_iter_gd = args.num_iter_gd
n_shot = args.n_shot
num_q = args.num_q # N # IMPORTANT 
Delta = args.Delta # g
layer = args.layer  # Define the number of layers in the quantum circuit
num_p = layer * 4  # Calculate the number of parameters
JOB_ID = args.JOB_ID
tol = args.tol
phys = args.phys

######################## problem setup ########################
xxz_op = xxz_hamiltonian(num_q, Jz=Delta, bc='periodic')

eigensolver = NumPyMinimumEigensolver()
result = eigensolver.compute_minimum_eigenvalue(operator=xxz_op)

ground_energy = result.eigenvalue.real
print(f"Ground state energy = {ground_energy}")

######################## loss function ########################

def expectation_loss(weights):
    return expectation_loss_grad(num_q,
                                 layer,
                                 weights,
                                 circuit=circuit_QAOA_XXZ,
                                 obs=xxz_op,
                                 )


def estimate_loss(weights, shots):
    return expectation_loss_grad(num_q,
                                 layer,
                                 weights,
                                 circuit=circuit_QAOA_XXZ,
                                 obs=xxz_op,
                                 shots=shots)
                                
######################## weights dict setup ########################


with open(f"find_omegas_XXZ_results/compact_omegas_q{num_q}_l{layer}.pkl", "rb") as f:
    weights_dict = pickle.load(f)

for j in range(num_p):
    # omegas = weights_dict[f'weights_{j}']['omegas']
    # opt_mse, interp_nodes, inverse_interp_matrix = optimal_interp_points(omegas)
    # weights_dict[f'weights_{j}']['opt_mse'] = opt_mse
    # weights_dict[f'weights_{j}']['interp_nodes'] = interp_nodes
    # weights_dict[f'weights_{j}']['inverse_interp_matrix'] = inverse_interp_matrix
    print(weights_dict[f'weights_{j}']['opt_mse'])


######################## Solver setup ########################

def main():
    # Check if the folder exists and delete it if it does.
    # Note that we delete everything inside the folder

    path = f'{JOB_ID}/{phys}'
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed existing directory: {path}")
    os.makedirs(path)

    all_data = []
    for exp_i in range(repeat):
        print('='*100)
        print(f'Experiment # {exp_i} begins.')

        # Define the initial value for x
        # Ensure that the initial point of each experiment exp_i is random
        initial_weights = np.random.uniform(0, 2*np.pi, size=num_p)

        # Initialize data_dict
        data_dict = {}

        ############################################################
        # Run OICD
        _, _, best_expvals, fun_calls, all_expvals, metrics = oicd(
            estimate_loss,
            expectation_loss,
            ground_energy,
            n_shot, weights_dict, initial_weights, num_iter_oicd,
            cyclic_mode=False,
            use_pratical_interp_flag=True,
            use_local_solvers_flag=False,
            use_global_solvers_flag = False,
            use_eigen_method_flag = True,
            # subproblem_method='BFGS',
            # subproblem_iter=None,
            use_exact_update_frequencey_1_flag = False,
            exact_mode=False, # for testing purpose, no noisy loss
            plot_flag=False,
            plot_argmin_flag = False,
            tol = tol,
        )


        data_dict['OICD'] = {
            'best_expvals': best_expvals,
            'fun_calls': fun_calls,
            'all_expvals': all_expvals,
            'metrics': metrics,
        }

        ############################################################
        # Run random coordinate descent
        _, best_expvals, fun_calls, all_expvals, metrics = rcd(
            estimate_loss,
            expectation_loss,
            ground_energy,
            n_shot, weights_dict, initial_weights, num_iter_rcd,
            cyclic_mode=False,
            learning_rate=lr_rcd,
            decay_step=30,
            decay_rate=-1,
            decay_threshold=1e-4,
            exact_mode=False,
            plot_flag=False,
            tol = tol,
        )

        data_dict['RCD'] = {
            'best_expvals': best_expvals,
            'fun_calls': fun_calls,
            'all_expvals': all_expvals,
            'metrics': metrics,
        }

        ############################################################
        # Run gradient descent
        _, best_expvals, fun_calls, all_expvals, metrics = gd(
            estimate_loss,
            expectation_loss,
            ground_energy,
            n_shot, weights_dict, initial_weights, num_iter_gd,
            learning_rate=lr_gd,
            exact_mode=False,
            plot_flag=False,
            tol = tol,
        )

        data_dict['GD'] = {
            'best_expvals': best_expvals,
            'fun_calls': fun_calls,
            'all_expvals': all_expvals,
            'metrics': metrics,
        }
        
        all_data.append(data_dict)

        ############################################################
        
        print()
        print('FINISHED')

    data_name = path + f'/data_dict.pkl'
    with open(data_name, 'wb') as file:
        pickle.dump(all_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    return path, JOB_ID, phys

if __name__ == "__main__":
    path, JOB_ID, phys = main()
    all_data = load(path + f'/data_dict.pkl')
    plot_results(all_data, path, JOB_ID, phys, x_right=600)

    