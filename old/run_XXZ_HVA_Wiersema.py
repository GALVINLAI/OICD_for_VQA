import argparse
import os, shutil

import numpy as np
from numpy.linalg import eigh

from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators import Operator
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit

from algo.utils import dump, make_dir
from algo.utils import mse, optimal_interp_points, interp_matrix

from algo.oicd import oicd
from algo.gd import gd
from algo.rcd import rcd

def create_parser():
    # Instantiate the argument parser
    parser = argparse.ArgumentParser(description="A simple argument parser")
    
    # Add number of qubits.
    parser.add_argument('--num_q', type=int, default=4, help='The number of qubits')

    # Parameters for the XXZ hamiltonian
    parser.add_argument('--Delta', type=float, default=0.5, 
                        help='Parameters for the XXZ hamiltonian')
    
    # Add number of layers of circuits.
    parser.add_argument('--layer', type=int, default=2, 
                        help='The number of layers of circuits.')

    # Add the repeat argument
    parser.add_argument('--repeat', type=int, default=1, 
                        help='The number of times to repeat the experiment')

    # Add the learning rate for gradient descent argument
    parser.add_argument('--lr_gd', type=float, default=0.01, 
                        help='The learning rate for the gradient descent')
    
    # Add the learning rate for random coordinate descent argument
    parser.add_argument('--lr_rcd', type=float, default=0.01, 
                        help='The learning rate for the random coordinate descent')
    
    # Add the number of iterations argument
    parser.add_argument('--num_iter_oicd', type=int, default=500, 
                    help='The number of iterations for the optimization algorithm [OICD]')
    parser.add_argument('--num_iter_rcd', type=int, default=500, 
                    help='The number of iterations for the optimization algorithm [RCD]')
    parser.add_argument('--num_iter_gd', type=int, default=50, 
                    help='The number of iterations for the optimization algorithm [GD]')

    parser.add_argument('--n_shot', type=int, default=1000, 
                    help='The number of shots for each evluation')
        
    return parser

args = create_parser().parse_args()

print("Run the HVA algorithm for the QAOA_XXZ_Wiersema model")
print(f"Number of qubits: {args.num_q}")
print(f"Delta value (Parameters of hamiltonian): {args.Delta}")
print(f"Number of layers of circuits: {args.layer}")
print(f"Number of shots: {args.n_shot}")
print(f"Number of parameter: {args.layer*4}")
print(f"Repeat count: {args.repeat}")
print(f"Gradient descent learning rate: {args.lr_gd}")
print(f"Random coordinate descent learning rate: {args.lr_rcd}")
print(f"Number of iterations, [OICD]: {args.num_iter_oicd}")
print(f"Number of iterations, [RCD]: {args.num_iter_rcd}")
print(f"Number of iterations, [GD]: {args.num_iter_gd}")

repeat = args.repeat
lr_gd = args.lr_gd
lr_rcd = args.lr_rcd
num_iter_oicd = args.num_iter_oicd
num_iter_rcd = args.num_iter_rcd
num_iter_gd = args.num_iter_gd
n_shot = args.n_shot

######################## max-cut problem setup ########################
num_q = args.num_q # N # IMPORTANT 
Delta = args.Delta # IMPORTANT

List_1 = []
op = ""
op = "X"
for k in range(num_q-2):
    op += "I"
op += "X"
List_1.append(op)
for i in range(num_q-1):
    op = ""
    for k in range(i):
        op += "I"
    op += "XX"
    for k in range(i+1,num_q-1):
        op += "I"
    List_1.append(op)
List_1.reverse()
H1 = SparsePauliOp(List_1, np.ones(num_q))

List_2 = []
op = "Y"
for k in range(num_q-2):
    op += "I"
op += "Y"
List_2.append(op)
for i in range(num_q-1):
    op = ""
    for k in range(i):
        op += "I"
    op += "YY"
    for k in range(i+1,num_q-1):
        op += "I"
    List_2.append(op)
List_2.reverse()
H2 = SparsePauliOp(List_2, np.ones(num_q))

H_first = SparsePauliOp.sum([H1,H2])

List_3 = []
op = "Z"
for k in range(num_q-2):
    op += "I"
op += "Z"
List_3.append(op)  
for i in range(num_q-1):
    op = ""
    for k in range(i):
        op += "I"
    op += "ZZ"
    for k in range(i+1,num_q-1):
        op += "I"
    List_3.append(op)

List_3.reverse()    
H_second = SparsePauliOp(List_3, Delta*np.ones(num_q))

# set H = H_first + H_second
H = SparsePauliOp.sum([H_first,H_second])
Hmat = Operator(H)
Hmat = Hmat.data # This is the matrix representation of the Hamiltonian

# # Print with detailed descriptions
# print(f"Number of qubits (num_q): {num_q}")
# print(f"Delta value: {Delta}")
# print("List_1 (Pauli terms for XX interactions):")
# print(List_1)
# print("List_2 (Pauli terms for YY interactions):")
# print(List_2)
# print("List_3 (Pauli terms for ZZ interactions):")
# print(List_3)
# print(H.size)
# print(H)

######################## ground state calculation ########################
# Compute eigenvalues and eigenvectors
e, v = eigh(Hmat)

# Identify the ground state (minimum eigenvalue)
ground_e = np.min(e)
min_index = np.argmin(e)
v_min = v[:, min_index]  # Ground state eigenvector

# Check degeneracy of the ground state
degeneracy = np.sum(np.isclose(e, ground_e))

# Print warning if ground state is not unique
if degeneracy > 1:
    print(f"Warning: Ground state is not unique. Degeneracy = {degeneracy}. Fidelity is not good metric.")
else:
    print("Ground state is unique.")
    
# Optional: Output for debugging
print(f"Ground state energy: {ground_e}")
print(f"Eigenvalues: {e}")


######################## circuit construction ########################

"""Circuit construction"""
layer = args.layer  # Define the number of layers in the quantum circuit

# This needs to be determined based on the circuit
num_p = layer * 4  # Calculate the number of parameters

weights = ParameterVector("w", num_p)  # Create a vector of parameters (parameters of the quantum circuit)

def circuit_QAOA_XXZ(weights):
    circ = QuantumCircuit(num_q, num_q)
    
    # Input layer to prepare Bell state -\psi
    for j in range(num_q):
        circ.x(j)  # Apply X gate to all qubits, initializing them from |0> to |1>.
    for j in range(int(num_q / 2)):
        # Then, for each pair of adjacent qubits (2j and 2j+1), apply H gate and CX (CNOT) gate to create Bell states.
        circ.h(2 * j)  # Apply Hadamard gate to each pair of qubits to create superposition states
        circ.cx(2 * j, 2 * j + 1)  # Create Bell state: Apply CNOT gate to each pair of qubits

    # QAOA Ansatz (variational layers)
    # Each layer consists of two parts: odd layers and even layers. The total number of layers is 'layer', with 2 sub-layers inside.
    for i in range(layer):
        ## Odd layers
        for j in range(int(num_q / 2)):
            circ.rzz(weights[4 * i], 2 * j + 1, (2 * j + 2) % num_q)  ## ZZ gates in odd sum
            # weights  [0]
        for j in range(int(num_q / 2)):
            circ.ryy(weights[4 * i + 1], 2 * j + 1, (2 * j + 2) % num_q)  ## YY gates in odd sum
            # weights  [1]
        for j in range(int(num_q / 2)):
            circ.rxx(weights[4 * i + 1], 2 * j + 1, (2 * j + 2) % num_q)  ## XX gates in odd sum
            # weights  [1]

        ## Even layers
        for j in range(int(num_q / 2)):
            circ.rzz(weights[4 * i + 2], 2 * j, 2 * j + 1)  ## ZZ gates in even sum
            # weights  [2]
        for j in range(int(num_q / 2)):
            circ.ryy(weights[4 * i + 3], 2 * j, 2 * j + 1)  ## YY gates in even sum
            # weights  [3]
        for j in range(int(num_q / 2)):
            circ.rxx(weights[4 * i + 3], 2 * j, 2 * j + 1)  ## XX gates in even sum
            # weights  [3]

    return circ 

# qc = circuit_QAOA_XXZ(weights)
# # print(qc)
# qc.draw("mpl")


######################## weights dict setup ########################

weights_dict = {}

# RZZ gates

if num_q == 4:
    omegas = [2]
elif num_q == 6:
    omegas = [2]
elif num_q == 8:
    omegas = [2, 4]
elif num_q == 10:
    omegas = [2, 4]
elif num_q == 12:
    omegas = [2, 4, 6]
elif num_q == 14:
    omegas = [2, 4, 6]

omegas_1 = omegas
# opt_mse, interp_nodes_1, inverse_interp_matrix_1 = optimal_interp_points(omegas_1)
interp_nodes_1 = np.linspace(0, 2 * np.pi, 2 * len(omegas_1) + 1, endpoint=False)
inverse_interp_matrix_1 = np.linalg.inv(interp_matrix(interp_nodes_1, omegas_1))
print(f"Minimum MSE: {mse(interp_nodes_1, omegas_1)}")

# RYY+RXX gates

if num_q == 4:
    omegas = [2, 4]
elif num_q == 6:
    omegas = [2, 4]
elif num_q == 8:
    omegas = [2, 4, 6, 8]
elif num_q == 10:
    omegas = [2, 4, 6, 8]
elif num_q == 12:
    omegas = [2, 4, 6, 8, 10, 12]
elif num_q == 14:
    omegas = [2, 4, 6, 8, 10, 12]

omegas_2 = omegas
# opt_mse, interp_nodes_2, inverse_interp_matrix_2 = optimal_interp_points(omegas_2)
interp_nodes_2 = np.linspace(0, 2 * np.pi, 2 * len(omegas_2) + 1, endpoint=False)
inverse_interp_matrix_2 = np.linalg.inv(interp_matrix(interp_nodes_2, omegas_2))
print(f"Minimum MSE: {mse(interp_nodes_2, omegas_2)}")


# Construct weights_dict
weights_dict = {}
for j in range(num_p): 
    if j % 2 == 0:  # Odd layers with RZZ gates
        weights_dict[f'weights_{j}'] = {
            'omegas': omegas_1,
            'scale_factor': 2.0,
            'interp_nodes': interp_nodes_1,
            'inverse_interp_matrix': inverse_interp_matrix_1,
        }
    elif j % 2 == 1:  # Even layers with RYY+RXX gates
        weights_dict[f'weights_{j}'] = {
            'omegas': omegas_2,
            'scale_factor': 2.0,
            'interp_nodes': interp_nodes_2,
            'inverse_interp_matrix': inverse_interp_matrix_2,
        }

######################## loss and fidelity function construction ########################
simulator = AerSimulator()

def estimate_loss(WEIGHTS, SHOTS):

    estimate_1 = 0
    estimate_2 = 0
    estimate_3 = 0

    ########################### XX
    qc = circuit_QAOA_XXZ(WEIGHTS)
    for i in range(num_q):
        qc.h(i)
    qc = transpile(qc, simulator)
    ind = list(range(num_q))
    rind = ind
    rind.reverse()
    qc.measure(ind, rind)
    result = simulator.run(qc, shots = SHOTS, memory=True).result()
    c = result.get_memory(qc) ## output distribution of 0 and 1

    for i in range(SHOTS):
        c_i = c[i]

        for j in range(num_q-1):
            if c_i[num_q-1-j] == c_i[num_q-1-(j+1)]:
                estimate_1 += 1
            else:
                estimate_1 += -1
        
        if c_i[num_q-1-0] == c_i[num_q-1-(num_q-1)]:
            estimate_1 += 1
        else:
            estimate_1 += -1

    estimate_1 = estimate_1/SHOTS

    ########################### YY
    qc = circuit_QAOA_XXZ(WEIGHTS)
    for i in range(num_q):
        qc.sdg(i)
        qc.h(i)
    qc = transpile(qc, simulator)
    ind = list(range(num_q))
    rind = ind
    rind.reverse()
    qc.measure(ind, rind)
    result = simulator.run(qc, shots = SHOTS, memory=True).result()
    c = result.get_memory(qc) ## output distribution of 0 and 1

    for i in range(SHOTS):
        c_i = c[i]

        for j in range(num_q-1):
            if c_i[num_q-1-j] == c_i[num_q-1-(j+1)]:
                estimate_2 += 1
            else:
                estimate_2 += -1
        
        if c_i[num_q-1-0] == c_i[num_q-1-(num_q-1)]:
            estimate_2 += 1
        else:
            estimate_2 += -1
            
    estimate_2 = estimate_2/SHOTS

    ########################### ZZ
    qc = circuit_QAOA_XXZ(WEIGHTS)
    qc = transpile(qc, simulator)
    ind = list(range(num_q))
    rind = ind
    rind.reverse()
    qc.measure(ind, rind)
    result = simulator.run(qc, shots = SHOTS, memory=True).result()
    c = result.get_memory(qc) ## output distribution of 0 and 1

    for i in range(SHOTS):
        c_i = c[i]

        for j in range(num_q-1):
            if c_i[num_q-1-j] == c_i[num_q-1-(j+1)]:
                estimate_3 += 1
            else:
                estimate_3 += -1
        
        if c_i[num_q-1-0] == c_i[num_q-1-(num_q-1)]:
            estimate_3 += 1
        else:
            estimate_3 += -1
            
    estimate_3 = estimate_3/SHOTS

    estimate = estimate_1 + estimate_2 + Delta*estimate_3

    return estimate


def expectation_loss(WEIGHTS):
    qc = circuit_QAOA_XXZ(WEIGHTS)
    qc.save_statevector()
    qc = transpile(qc, simulator)
    result = simulator.run(qc).result()
    state_vector = result.get_statevector(qc)
    psi = np.asarray(state_vector)
    # ==========================================================================
    Hpsi = Hmat.dot(psi)
    expectation = np.inner(np.conjugate(psi),Hpsi)
    return np.real(expectation)


def fidelity(WEIGHTS):
    qc = circuit_QAOA_XXZ(WEIGHTS)
    qc.save_statevector()
    qc = transpile(qc, simulator)
    result = simulator.run(qc).result()
    state_vector = result.get_statevector(qc)
    psi = np.asarray(state_vector)
    # ==========================================================================
    return np.absolute(np.vdot(psi,v_min))**2


def energy_ratio(WEIGHTS):
    return np.abs(expectation_loss(WEIGHTS) / ground_e)

######################## Solver setup ########################

def main():
    # Check if the folder exists and delete it if it does.
    # Note that we delete everything inside the folder
    make_dir('exp/QAOA_XXZ_Wiersema')

    dir_path = f'exp/QAOA_XXZ_Wiersema/'
    
    # if os.path.exists(dir_path):
    #     shutil.rmtree(dir_path)
    #     print(f"Removed existing directory: {dir_path}")

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
        final_weights_oicd, best_expected_record_value_oicd, best_fid_record_value_oicd, func_oicd, every_expected_oicd, every_fid_oicd  = oicd(
            estimate_loss,
            expectation_loss,
            fidelity,
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
        )

        data_dict.update({
            'x_oicd': final_weights_oicd,
            'best_expected_oicd': best_expected_record_value_oicd / ground_e,
            'fid_oicd': best_fid_record_value_oicd,
            'func_oicd': func_oicd,
            'every_expected_oicd': every_expected_oicd / ground_e,
            'every_fid_oicd': every_fid_oicd
        })

        ############################################################
        # Run random coordinate descent
        final_weights_rcd, best_expected_record_value_rcd, best_fid_record_value_rcd, func_rcd, every_expected_rcd, every_fid_rcd = rcd(
            estimate_loss,
            expectation_loss,
            fidelity,
            n_shot, weights_dict, initial_weights, num_iter_rcd,
            cyclic_mode=False,
            learning_rate=lr_rcd,
            decay_step=10,
            decay_rate=-1,
            decay_threshold=1e-4,
            exact_mode=False,
            plot_flag=False,
        )

        data_dict.update({
            'x_rcd': final_weights_rcd,
            'best_expected_rcd': best_expected_record_value_rcd / ground_e,
            'fid_rcd': best_fid_record_value_rcd,
            'func_rcd': func_rcd,
            'every_expected_rcd': every_expected_rcd / ground_e,
            'every_fid_rcd': every_fid_rcd
        })

        ############################################################
        # Run gradient descent
        final_weights_gd, best_expected_record_value_gd, best_fid_record_value_gd, func_gd, every_expected_gd, every_fid_gd = gd(
            estimate_loss,
            expectation_loss,
            fidelity,
            n_shot, weights_dict, initial_weights, num_iter_gd,
            learning_rate=lr_gd,
            exact_mode=False,
            plot_flag=False,
        )

        data_dict.update({
            'x_gd': final_weights_gd,
            'best_expected_gd': best_expected_record_value_gd / ground_e,
            'fid_gd': best_fid_record_value_gd,
            'func_gd': func_gd,
            'every_expected_gd': every_expected_gd / ground_e,
            'every_fid_gd': every_fid_gd
        })
        
        ############################################################
        make_dir(f'exp/QAOA_XXZ_Wiersema/N{args.num_q}/exp_{exp_i}')
        dump(data_dict, f'exp/QAOA_XXZ_Wiersema/N{args.num_q}/exp_{exp_i}/data_dict.pkl')
        dump(weights_dict, f'exp/QAOA_XXZ_Wiersema/N{args.num_q}/exp_{exp_i}/weights_dict.pkl')
        
        print()
        print('FINISHED')

if __name__ == "__main__":
    main()
