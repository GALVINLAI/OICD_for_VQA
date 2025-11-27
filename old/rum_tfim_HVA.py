import os
import numpy as np
from algo.oicd import oicd
from algo.rcd import rcd
from algo.gd import gd

from qiskit_algorithms import NumPyMinimumEigensolver
from algo.utils import (
    tfim_hamiltonian,
    circuit_HVA_TIFM,
    expectation_loss_grad,
    interp_matrix,
    mse,
)

def make_weights_dict(num_p):
    weights_dict = {}
    omegas_1 = [2]
    interp_nodes_1 = np.linspace(0, 2 * np.pi, 2 * len(omegas_1) + 1, endpoint=False)
    inverse_interp_matrix_1 = np.linalg.inv(interp_matrix(interp_nodes_1, omegas_1))

    omegas_2 = [2]
    interp_nodes_2 = np.linspace(0, 2 * np.pi, 2 * len(omegas_2) + 1, endpoint=False)
    inverse_interp_matrix_2 = np.linalg.inv(interp_matrix(interp_nodes_2, omegas_2))

    for j in range(num_p): 
        if j % 2 == 0:
            weights_dict[f'weights_{j}'] = {
                'omegas': omegas_1,
                'scale_factor': 2.0,
                'interp_nodes': interp_nodes_1,
                'inverse_interp_matrix': inverse_interp_matrix_1,
            }
        else:
            weights_dict[f'weights_{j}'] = {
                'omegas': omegas_2,
                'scale_factor': 2.0,
                'interp_nodes': interp_nodes_2,
                'inverse_interp_matrix': inverse_interp_matrix_2,
            }
    return weights_dict

def run_algorithms(num_q, repeat=3, n_shot=1024, Delta=0.5, tol=1e-2):
    layer = int(num_q / 2)
    num_p = layer * 2
    tfim_op = tfim_hamiltonian(num_q, J=-1.0, g=-1 * Delta, bc='periodic')
    ground_energy = NumPyMinimumEigensolver().compute_minimum_eigenvalue(tfim_op).eigenvalue.real

    weights_dict = make_weights_dict(num_p)

    fun_calls_dict = {'OICD': [], 'RCD': [], 'GD': []}

    for i in range(repeat):
        initial_weights = np.random.uniform(0, 2 * np.pi, size=num_p)

        # OICD
        success, _, fun_calls, _, _ = oicd(
            lambda w, n_shot: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op, shots=n_shot),
            lambda w: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op),
            ground_energy,
            n_shot,
            weights_dict,
            initial_weights,
            num_iter=500,
            cyclic_mode=False,
            use_pratical_interp_flag=True,
            use_local_solvers_flag=False,
            use_global_solvers_flag=False,
            use_eigen_method_flag=True,
            use_exact_update_frequencey_1_flag=False,
            exact_mode=False,
            plot_flag=False,
            plot_argmin_flag=False,
            tol=tol,
        )
        # if success:
        #     fun_calls_dict['OICD'].append(fun_calls[-1])
        # else:
        #     fun_calls_dict['OICD'].append("F")

        fun_calls_dict['OICD'].append(fun_calls[-1])

        # RCD
        success, _, fun_calls, _, _ = rcd(
            lambda w, n_shot: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op, shots=n_shot),
            lambda w: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op),
            ground_energy,
            n_shot,
            weights_dict,
            initial_weights,
            num_iter=500,
            cyclic_mode=False,
            learning_rate=0.02,
            decay_step=30,
            decay_rate=-1,
            decay_threshold=1e-4,
            exact_mode=False,
            plot_flag=False,
            tol=tol,
        )
        # if success:
        #     fun_calls_dict['RCD'].append(fun_calls[-1])
        # else:
        #     fun_calls_dict['RCD'].append("F")
        fun_calls_dict['RCD'].append(fun_calls[-1])

        # GD
        success, _, fun_calls, _, _ = gd(
            lambda w, n_shot: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op, shots=n_shot),
            lambda w: expectation_loss_grad(num_q, layer, w, circuit_HVA_TIFM, tfim_op),
            ground_energy,
            n_shot,
            weights_dict,
            initial_weights,
            num_iter=int(1000 / (2 * layer * 2) + 1),
            learning_rate=0.01,
            exact_mode=False,
            plot_flag=False,
            tol=tol,
        )

        # if success:
        #     fun_calls_dict['GD'].append(fun_calls[-1])
        # else:
        #     fun_calls_dict['GD'].append("F")

        fun_calls_dict['RCD'].append(fun_calls[-1])

    return {
        algo: (np.mean(calls) if calls else float('inf'))
        for algo, calls in fun_calls_dict.items()
    }

def main():
    print(f"{'num_q':<6} {'OICD':>10} {'RCD':>10} {'GD':>10}")
    print("-" * 40)
    for num_q in range(4, 7):
        results = run_algorithms(num_q)
        print(f"{num_q:<6} {results['OICD']:>10.1f} {results['RCD']:>10.1f} {results['GD']:>10.1f}")

if __name__ == "__main__":
    main()
