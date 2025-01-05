
import numpy as np
import matplotlib.pyplot as plt
import random

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


def plot_univariate_function(true_cost_fun, indices_to_check, weights, x_range=(-10, 10), num_points=100):
    """
    Plot the univariate functions that vary one or more weights at `indices_to_check`.

    Args:
    - true_cost_fun: The true (no noise) cost function to evaluate (takes `weights` as input).
    - indices_to_check: A list of indices in `weights` where we will vary the weight to check the function's behavior.
    - weights: The array of weights for the cost function.
    - x_range: The range of `x` values to plot the function over. Default is (-10, 10).
    - num_points: The number of points to evaluate in the plot. Default is 100.

    This function will plot multiple univariate functions, each varying one weight at a given index.
    """

    # Number of subplots (one per index in `indices_to_check`)
    num_plots = len(indices_to_check)
    
    # Create subplots with enough space
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one plot
    
    # Loop over each index to create a univariate function plot
    for i, j in enumerate(indices_to_check):
        # Define the univariate function that varies the weight at position `j`
        univariate_fun = lambda x: true_cost_fun(np.concatenate([weights[:j], [x], weights[j+1:]]))

        # Generate a range of x values for plotting
        x_vals = np.linspace(x_range[0], x_range[1], num_points)

        # Compute the corresponding y values for the univariate function
        y_vals = np.array([univariate_fun(x) for x in x_vals])

        # Plot the univariate function
        axes[i].plot(x_vals, y_vals, label=f'Univariate function (index {j})')
        axes[i].set_xlabel(f'Weight at index {j}')
        axes[i].set_ylabel('Cost Function Value')
        axes[i].set_title(f'Univariate Function Behavior for index {j}')
        axes[i].grid(True)
        axes[i].legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()




def parameter_shift_for_equidistant_frequencies(estimate_loss, weights, index, omegas, n_shot):
    r = len(omegas)
    x_mus = [(2 * mu - 1) * np.pi / (2 * r) for mu in range(1, 2 * r + 1)]
    
    # Compute the coefficients
    coefs = np.array([(-1) ** (mu - 1) / (4 * r * np.sin(x_mus[mu - 1] / 2) ** 2) for mu in range(1, 2 * r + 1)])
    
    x_bar = weights[index]  # Get the current parameter value
    evals = []
    
    # Perform the parameter shift
    for mu in range(1, 2 * r + 1):
        # Create a copy of weights to avoid modifying the original weights
        new_weights = weights.copy()
        new_weights[index] = x_bar + x_mus[mu - 1]
        
        # Compute the loss
        evals.append(estimate_loss(new_weights, n_shot))
    
    # Sum the product of coefficients and computed losses
    return np.sum(coefs * np.array(evals))    

# List of Pauli operators as strings
pauli_operators = ['I', 'X', 'Y', 'Z']

def random_pauli_word_string(n):
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
