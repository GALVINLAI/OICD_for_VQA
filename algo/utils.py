import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import differential_evolution
import os
import pickle
import random
import re
from functools import reduce

# ================================================================
#          check utils
# ================================================================

def compare_functions(func1, func2, num_tests=50, input_range=(-10 * np.pi, 10 * np.pi), atol=1e-8, rtol=1e-5):
    """
    Compare the behavior of two functions
    - func1, func2: The two functions to compare
    - num_tests: The number of tests to perform
    - input_range: The range for random inputs (default is -10π to 10π)
    - atol: Absolute tolerance
    - rtol: Relative tolerance
    """
    # Print what we are doing at the beginning
    # print(f"Comparing the behavior of two functions over {num_tests} random inputs in the range {input_range}.")

    num_consistent = 0  # Count for consistent test points
    num_failed = 0  # Count for failed test points
    failed_tests = []  # Store failed inputs and their results

    for _ in range(num_tests):
        # Generate random input
        x = np.random.uniform(input_range[0], input_range[1])
        
        # Get the output from both functions
        output1 = func1(x)
        output2 = func2(x)
        
        # Compare if both outputs are close to each other
        if np.isclose(output1, output2, atol=atol, rtol=rtol):
            num_consistent += 1  # Increment if results are consistent
        else:
            num_failed += 1  # Increment if results differ
            failed_tests.append((x, output1, output2))  # Record failed inputs and their results
    
    # Output comparison details
    # print(f"Testing range: {input_range}")
    # print(f"Total tests: {num_tests}")
    print(f"Consistent results: {num_consistent}/{num_tests}")
    
    if num_failed > 0:
        print(f"Failed tests: {num_failed}/{num_tests}")
        for x, output1, output2 in failed_tests:
            print(f"  At input {x}, func1 returned {output1}, func2 returned {output2}")
    else:
        # print("【All tests passed within the given tolerance！Two function are the same!】")
        print("【All Passed】")
    
    return num_failed == 0  # Return True if there are no failed tests


# Interpolation matrix generation function
def interp_matrix(interp_points, Omegas):
    r = len(Omegas)
    return np.array([[1/np.sqrt(2)] + [func(Omegas[k] * x) for k in range(r) for func in (np.cos, np.sin)] for x in interp_points])


def check_is_trigometric(true_cost_fun, index_to_check, omegas, weights, opt_interp_flag=True):
    """
    Check if a cost function behaves like a trigonometric function.
    This only for the equdistant frequency case.

    Args:
    - true_cost_fun: The true (no noise) cost function to evaluate (takes `weights` as input).
    - index_to_check: The index in `weights` where we will vary the weight to check the function's behavior.
    - omegas: A list or array of equidistant frequencies.
    - weights: The array of weights for the cost function.
    - opt_interp_flag: Flag indicating whether to use optimal or random interpolation points. Default is True for optimal.

    This function performs the following:
    - Varies the weight at position `index_to_check` and evaluates the cost function.
    - Computes the coefficients using optimal interpolation (or random interpolation, depending on `opt_interp_flag`).
    - Defines and compares two functions: the univariate function (where only one weight is varied) and the estimated trigonometric function.
    """

    # The index in the weights array where we will vary the value
    j = index_to_check

    # Define a univariate function that varies the weight at position j
    # This function is used to evaluate how the cost function behaves when the weight at index `j` changes
    univariate_fun = lambda x: true_cost_fun(np.concatenate([weights[:j], [x], weights[j+1:]]))

    # Length of the omegas array (number of frequencies)
    r = len(omegas)

    # If opt_interp_flag is True, use optimal interpolation points spaced 2π/(2*r+1) over [0, 2π]
    # Otherwise, use random interpolation points in the range [0, 2π]
    if opt_interp_flag: 
        interp_points = np.linspace(0, 2 * np.pi, 2*r + 1, endpoint=False)  # Optimal interpolation points
    else:  # Random interpolation points
        interp_points = np.random.uniform(0, 2 * np.pi, 2*r + 1)  

    # List to store function values at the interpolation points
    fun_vals = []
    
    # Evaluate the true cost function at the interpolation points
    for point in interp_points:
        weights[j] = point  # Set the weight at index `j` to the current interpolation point
        fun_val = true_cost_fun(weights)  # Evaluate the cost function with the modified weights
        fun_vals.append(fun_val)  # Store the result

    # Convert the function values to a numpy array
    fun_vals = np.array(fun_vals)
    
    # Create an interpolation matrix based on the interpolation points and frequencies (omegas)
    # reg_param=1e-8
    opt_interp_matrix = interp_matrix(interp_points, omegas) # + reg_param * np.eye(2*r + 1)

    # Solve the system of linear equations to estimate the coefficients (hat_z)
    # This solves the equation: opt_interp_matrix * hat_z = fun_vals
    hat_z = np.linalg.solve(opt_interp_matrix, fun_vals)

    # Define the estimated function (hat_f) based on the computed coefficients (hat_z)
    # This function approximates the true function using a trigonometric basis (cosine and sine)
    def hat_f(x):  # Approximation function using the estimated coefficients
        # t_x is the vector of trigonometric terms (1, cos(omegas * x), sin(omegas * x))
        t_x = np.array([1 / np.sqrt(2)] + 
                       [func(omegas[k] * x).item() for k in range(r) for func in (np.cos, np.sin)])
        return np.inner(t_x, hat_z)  # Dot product with the estimated coefficients

    
    # Compare the univariate function (where only one weight is varied) with the estimated function
    compare_functions(univariate_fun, hat_f)

    # Print the estimated coefficients (hat_z)
    print("Estimated coefficients: ", hat_z)

    return hat_z


# ================================================================
#          algo utils
# ================================================================


# Mean Squared Error (MSE) function
def mse(interp_points, Omegas):
    # Create interpolation matrix
    A = interp_matrix(interp_points, Omegas)
    
    # Regularize the matrix
    regularized_matrix = A.T @ A + 1e-6 * np.eye(A.shape[1])  # Add a small regularization term to avoid singular matrix
    
    # Return the trace of the inverse of the matrix
    return np.trace(np.linalg.inv(regularized_matrix))


# Define the optimization function
def optimal_interp_points(Omegas): 
    r = len(Omegas)

    # Define the bounds for the interpolation points
    # bounds = [(-1e6, 1e6) for _ in range(2 * r + 1)]  # This is a loose boundary range
    bounds = [(0, 2*np.pi) for _ in range(2 * r + 1)]

    # Use differential evolution for optimization
    result_mse = differential_evolution(mse, bounds, args=(Omegas,), strategy='best1bin', maxiter=1000)

    opt_mse = result_mse.fun

    # Get the optimized interpolation points
    opt_interp_points = sorted(result_mse.x)

    # Return the optimized interpolation points and the inverse of the corresponding interpolation matrix
    return opt_mse, opt_interp_points, np.linalg.inv(interp_matrix(opt_interp_points, Omegas))


def parameter_shift_for_equidistant_frequencies(estimate_loss, weights, index, omegas, factor=1.0):
    # r = max(omegas)
    # r = len(omegas)

    r = len(omegas)

    x_mus = [(2 * mu - 1) * np.pi / (2 * r) for mu in range(1, 2 * r + 1)]
    
    # Compute the coefficients
    coefs = np.array([(-1) ** (mu - 1) / (4 * r * np.sin(x_mus[mu - 1] / 2) ** 2) for mu in range(1, 2 * r + 1)])
    
    x_bar = weights[index] * factor  # Get the current parameter value
    evals = []
    
    # Perform the parameter shift
    for mu in range(1, 2 * r + 1):
        # Create a copy of weights to avoid modifying the original weights
        new_weights = weights.copy()
        new_weights[index] = (x_bar + x_mus[mu - 1]) / factor
        
        # Compute the loss
        evals.append(estimate_loss(new_weights))
    
    # Sum the product of coefficients and computed losses
    return np.sum(coefs * np.array(evals)) * factor


def plot_every_iteration(expected_record_value, fidelity_record_value, name, approx_record_value=[]):
    
    clear_output(wait=True)

    # Create a 1x2 subplot
    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Approx Loss and True Loss on the first subplot
    # if approx_record_value is not None:
    if len(approx_record_value) > 0:
        axs[0].plot(approx_record_value, label='Approx Loss')
    axs[0].plot(expected_record_value, label='True Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_title(f'{name} Loss')
    axs[0].legend(fontsize=12)

    # Plot Fidelity on the second subplot
    axs[1].plot(fidelity_record_value, label='Fidelity', color='g')
    axs[1].axhline(y=1, color='r', linestyle='--', label='1')
    axs[1].set_xlabel('Iteration')
    axs[1].set_title('Fidelity')
    axs[1].legend(fontsize=12)

    # Show the plot
    plt.tight_layout()  # Automatically adjust the spacing between subplots
    plt.show()


def filter_omegas(estimated_coefficients, omegas, threshold=1e-8):
    # Skip the first coefficient if necessary (according to your example)
    estimated_coefficients = estimated_coefficients[1:]

    # Pair the coefficients
    pairs = [(estimated_coefficients[i], estimated_coefficients[i+1]) for i in range(0, len(estimated_coefficients), 2)]

    # Initialize a list to store the indices to remove
    indices_to_remove = []

    # Iterate through each pair and check the condition
    for i, (coef1, coef2) in enumerate(pairs):
        if abs(coef1) < threshold and abs(coef2) < threshold:
            indices_to_remove.append(i)

    # Remove corresponding omegas
    omegas_filtered = np.delete(omegas, indices_to_remove)

    return omegas_filtered


# ================================================================
#          files utils
# ================================================================

def make_dir(path):
    """
    Create a new directory if it does not exist.

    Parameters:
    path (str): The path of the directory to create.
    """
    # Check if the directory at the given path exists
    if not os.path.exists(path):
        # If the directory does not exist, create the directory
        os.makedirs(path, exist_ok=True)
        # The parameter exist_ok=True indicates that if the directory already exists, no error will be raised, and the operation will be simply ignored.


def load(filename):
    """
    Load data from a pickle file.

    Parameters:
    filename (str): The name of the pickle file.

    Returns:
    loaded: The data loaded from the pickle file.
    """
    # Ensure the file is a pickle file to prevent loading errors or incompatible file types.
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'rb') as file:
        # The 'rb' parameter indicates opening the file in binary read mode, which is necessary for reading pickle files.
        # Use pickle.load to deserialize the object from the file.
        loaded = pickle.load(file)
        
    return loaded


def dump(content, filename):
    """
    Save data to a pickle file.

    Parameters:
    content : The data to be saved.
    filename (str): The name of the pickle file.
    """
    # Ensure the file extension is '.pkl' to maintain file format consistency.
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'wb') as file:
        # The 'wb' parameter indicates opening the file in binary write mode, which is necessary for writing pickle files.
        pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)
        # The parameter protocol=pickle.HIGHEST_PROTOCOL specifies using the highest protocol version to ensure the serialized data can be compatible with future Python versions.

# ================================================================
#          other utils
# ================================================================


def random_pauli_word_string(n):

    # List of Pauli operators as strings
    pauli_operators = ['I', 'X', 'Y', 'Z']

    # Randomly choose a Pauli operator for each qubit
    pauli_word = ''.join(random.choice(pauli_operators) for _ in range(n))
    
    return pauli_word


def find_pauli_indices(pauli_word):
    # Initialize empty lists for X, Y, Z indices
    x_indices = []
    y_indices = []
    z_indices = []
    length = len(pauli_word)
    
    # Iterate through the string and track the indices of X, Y, and Z
    for i, char in enumerate(pauli_word):
        if char == 'X':
            x_indices.append(length-1-i)
        elif char == 'Y':
            y_indices.append(length-1-i)
        elif char == 'Z':
            z_indices.append(length-1-i)
    
    return x_indices, y_indices, z_indices


def process_hamiltonian_Zs(ham_str, num_q, max_stars):
    # Step 1: Split the Hamiltonian string by the minus '-' sign and remove leading/trailing spaces
    split_by_minus = list(map(str.strip, ham_str.split('-')))

    # Step 2: Rejoin the terms with '+-' to handle the minus sign in a unified manner
    modified_ham_str = '+-'.join(split_by_minus)

    # Step 3: Split the string by the plus '+' sign and remove leading/trailing spaces
    split_by_plus = list(map(str.strip, modified_ham_str.split('+')))

    # Step 4: Filter out any empty strings (if any) and get all valid terms
    all_terms = list(filter(None, split_by_plus))

    # Function to categorize terms by the number of '*' they contain, with max_stars limit
    def categorize_terms_by_stars(term_list, max_stars):
        star_categories = {}

        # Loop over the terms and count the number of '*' in each term
        for term in term_list:
            star_count = term.count('*')
            if star_count not in star_categories:
                star_categories[star_count] = []
            star_categories[star_count].append(term)

        # Limit the categorization to max_stars categories (i.e., 0 to max_stars)
        categorized_terms = [star_categories.get(i, []) for i in range(max_stars + 1)]
        return categorized_terms

    # Function to replace positions in a list with 'Z' based on terms starting with 'z'
    def replace_positions(input_list, num_q):
        result = ['I'] * num_q  # Initialize the result list with 'I' (identity operators)
        # Loop through the input list and replace with 'Z' if term starts with 'z'
        for item in input_list:
            if item.lower().strip().startswith('z'):
                position = int(item.strip()[1:])  # Extract the position number after 'z'
                if 0 <= position < num_q:
                    result[position] = 'Z'  # Replace corresponding position with 'Z'
        return ''.join(result)

    # Function to process each term and return its coefficient and operator positions
    def coef_ops(term, num_q):
        coef = 1.0  # Initialize coefficient to 1
        ops = []  # Initialize the list of operators as empty

        # Split the term by "*" and process each component
        for component in term.split("*"):
            try:
                coef *= float(component)  # Try converting the component to a float (coefficient)
            except ValueError:
                ops.append(component)  # If it's not a number, it must be an operator (like 'z0')

        # Return the coefficient and the operator positions as a string (e.g., 'ZZIZ')
        return coef, replace_positions(ops, num_q)

    # Function to process all terms in a given list and return their operator positions and coefficients
    def process_terms(term_list, num_q):
        term_positions = []  # List to store operator positions (e.g., 'Z', 'I')
        term_coeffs = []  # List to store coefficients of the terms

        # Loop through each term, calculate its coefficient and operator positions
        for term in term_list:
            coef, positions = coef_ops(term, num_q)
            term_positions.append(positions)
            term_coeffs.append(coef)

        return term_positions, term_coeffs

    # Step 1: Categorize terms by the number of stars, with a limit on max_stars
    categorized_terms = categorize_terms_by_stars(all_terms, max_stars)

    # Step 2: Process the terms in each category dynamically (no need to manually define categories)
    all_lists = []  # To hold the results for each star category (positions)
    all_coeffs = []  # To hold the coefficients for each star category

    # Loop through each star count (from 0 to max_stars)
    for i in range(max_stars + 1):
        terms_in_category = categorized_terms[i]
        if terms_in_category:  # If there are terms in this category
            term_positions, term_coeffs = process_terms(terms_in_category, num_q)
            all_lists.append(term_positions)
            all_coeffs.append(term_coeffs)
        else:
            all_lists.append([])  # If no terms for this category, add an empty list
            all_coeffs.append([])

    # Return all processed results as a list of tuples (positions, coefficients)
    return all_lists, all_coeffs
    

def find_pauli_indices(pauli_word):
    # Initialize empty lists for X, Y, Z indices
    x_indices = []
    y_indices = []
    z_indices = []
    length = len(pauli_word)
    
    # Iterate through the string and track the indices of X, Y, and Z
    for i, char in enumerate(pauli_word):
        if char == 'X':
            x_indices.append(length-1-i)
        elif char == 'Y':
            y_indices.append(length-1-i)
        elif char == 'Z':
            z_indices.append(length-1-i)
    
    return x_indices, y_indices, z_indices



# Define Pauli matrices and identity matrix
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)

# Create a mapping from string to corresponding matrix
MATRIX_MAP = {'x': X, 'y': Y, 'z': Z, 'i': I}

# Example to compute the matrix representation of X0*Y1*Z2, where X0 denotes applying Pauli-X matrix to the 0th qubit,
# Y1 denotes applying Pauli-Y matrix to the 1st qubit, and Z2 denotes applying Pauli-Z matrix to the 2nd qubit
def term_to_matrix(term, n_qubits):
    """Convert a term to its matrix representation."""
    # Initialize list with identity operators
    operators = [I] * n_qubits

    for component in term.split("*"):
        # Split the term by asterisk * into multiple components. For example, 'X0*Y1*Z2' is split into ['X0', 'Y1', 'Z2'].
        component = component.strip()
        if not component:  # Skip if the component is empty.
            continue
        operator = component[0].lower()  # Extract the operator (e.g., 'x', 'y', 'z', 'i') and convert to lowercase.
        qubit = int(component[1:])  # Extract the qubit number on which the operator acts.
        operators[qubit] = MATRIX_MAP[operator]  # Replace the identity matrix at the corresponding position with the operator matrix.

    # Compute the tensor product of the operators
    return reduce(np.kron, operators)  # Compute the tensor product X ⊗ Y ⊗ Z to get the final matrix representation.
    # reduce is a function in Python, imported from the functools module, used for cumulative computation on elements in a sequence.
    # The working principle of reduce is to apply the specified binary function to the first two elements of the sequence,
    # then apply the result to the next element, and so on until all elements in the sequence are processed.


def hamiltonian_to_matrix(hamiltonian_str, n_qubits=-1):
    """
    Convert a Hamiltonian to its matrix representation.
    """

    # Handle the minus signs in the string, converting them to addition terms with a negative sign.
    hamiltonian_str = '+-'.join(map(str.strip, hamiltonian_str.split('-')))
    # hamiltonian_str.split('-'): Split the original Hamiltonian string by minus signs '-', generating a list of substrings.
    # For example, if the original string is "-3.0 + 0.5 * Z0 * Z2 - 0.25 * Z1 * Z0", it will be split into ['-3.0 + 0.5 * Z0 * Z2 ', ' 0.25 * Z1 * Z0'].

    # map(str.strip, ...): Apply the strip method to each substring after splitting to remove leading and trailing whitespace.
    # Continuing the previous example, it becomes ['-3.0 + 0.5 * Z0 * Z2', '0.25 * Z1 * Z0'].

    # '+-'.join(...): Reconnect the processed substrings with '+-'.
    # This step makes each minus sign explicit as part of a negative number.
    # For example, the final result will be '-3.0 + 0.5 * Z0 * Z2 +- 0.25 * Z1 * Z0'.

    terms = list(filter(None, map(str.strip, hamiltonian_str.split('+'))))
    # Split the Hamiltonian string by plus signs '+' to generate a list of substrings.
    # map(str.strip, ...): Apply the strip method to each substring after splitting to remove leading and trailing whitespace.
    # Continuing the previous example, it becomes ['-3.0', '0.5 * Z0 * Z2', '- 0.25 * Z1 * Z0'].

    # list(filter(None, ...)): Filter out empty strings.
    # Although empty strings are not generated in this particular example, this is a safety measure in case there are extraneous plus signs or spaces in the input string.

    if n_qubits < 0:  # This code aims to automatically calculate the number of qubits involved when the user does not specify it.
        # If n_qubits is less than 0, it means the user did not specify the number of qubits, and it needs to be calculated automatically.
        # Get the number of qubits involved
        qubit_indices = [int(qubit[1:]) for term in terms for qubit in re.findall(r'[x|y|z|X|Y|Z|i|I]\d+', term)]
        # for term in terms: This is a loop that iterates through each element in the terms list.
        # for qubit in ...: This is a nested loop that further iterates over all qubit representations matching the regular expression in each term.

        # Extract the indices of the qubits involved in all terms. The slice qubit[1:] considers multi-digit numbers, hence the use of 1:.
        n_qubits = max(qubit_indices) + 1

    # Initialize the Hamiltonian matrix
    H = np.zeros((2**n_qubits, 2**n_qubits))

    for term in terms:
        coef = 1.0
        ops = []  # Initialize the operator list as empty.
        for component in term.split("*"):  # Split the term by asterisk * into multiple components.
            try:
                coef *= float(component)
                # Try to convert the component to a float and multiply it into the coefficient.
                # If the component is a number, this step will succeed and update the coefficient.
            except ValueError:
                # If the component is not a number (i.e., it's an operator), add it to the operator list ops.
                ops.append(component)
        # Add the term to the Hamiltonian matrix
        # The term_to_matrix function converts a single term to its matrix representation.
        # Convert the operator list to its matrix representation, multiply by the coefficient, and add to the Hamiltonian matrix H.
        H += coef * term_to_matrix("*".join(ops), n_qubits)

    return H

# Test the function
# hamiltonian_str = "-3.0 + 0.5 * Z0 + 0.25 * Z1 + 0.25 * Z2 + 0.5 * Z3 + 0.75 * Z0*Z2 - 0.25 * Z1*Z2 + 0.25 * Z0*Z1 + 0.25 * Z0*Z3 + 0.75 * Z1*Z3 + 0.25 * Z2*Z3 - 0.25 * Z0*Z1*Z2 - 0.25 * Z1*Z2*Z3"
# print(hamiltonian_to_matrix(hamiltonian_str).astype(np.float64))

