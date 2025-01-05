import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.optimize import differential_evolution
import os
import pickle

# def parameter_shift_for_equidistant_frequencies(estimate_loss, weights, index, R):
#     # r = max(omegas)
#     # r = len(omegas)

#     r = R
#     x_mus = [(2 * mu - 1) * np.pi / (2 * r) for mu in range(1, 2 * r + 1)]
    
#     # Compute the coefficients
#     coefs = np.array([(-1) ** (mu - 1) / (4 * r * np.sin(x_mus[mu - 1] / 2) ** 2) for mu in range(1, 2 * r + 1)])
    
#     x_bar = weights[index]  # Get the current parameter value
#     evals = []
    
#     # Perform the parameter shift
#     for mu in range(1, 2 * r + 1):
#         # Create a copy of weights to avoid modifying the original weights
#         new_weights = weights.copy()
#         new_weights[index] = x_bar + x_mus[mu - 1]
        
#         # Compute the loss
#         evals.append(estimate_loss(new_weights))
    
#     # Sum the product of coefficients and computed losses
#     return np.sum(coefs * np.array(evals))    


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
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

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



# Interpolation matrix generation function
def interp_matrix(interp_points, Omegas):
    r = len(Omegas)
    return np.array([[1/np.sqrt(2)] + [func(Omegas[k] * x) for k in range(r) for func in (np.cos, np.sin)] for x in interp_points])

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


def make_dir(path):
    """
    Create a new directory if it does not exist.

    Parameters:
    path (str): The path of the directory to create.
    """
    if not os.path.exists(path):
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
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'rb') as file:
        # The 'rb' parameter indicates opening the file in binary read mode, which is necessary for reading pickle files.
        loaded = pickle.load(file)
        
    return loaded

def dump(content, filename):
    """
    Save data to a pickle file.

    Parameters:
    content : The data to be saved.
    filename (str): The name of the pickle file.
    """
    assert filename.endswith('.pkl'), "File must be a '.pkl' file"
    
    with open(filename, 'wb') as file:
        # The 'wb' parameter indicates opening the file in binary write mode, which is necessary for writing pickle files.
        pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)
        # The parameter protocol=pickle.HIGHEST_PROTOCOL specifies using the highest protocol version to ensure the serialized data can be compatible with future Python versions.


def interp_matrix(interp_points, Omegas):
    r = len(Omegas)
    return np.array([[1/np.sqrt(2)] + [func(Omegas[k] * x) for k in range(r) for func in (np.cos, np.sin)] for x in interp_points])

