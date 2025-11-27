import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from scipy.optimize import minimize
from IPython.display import clear_output
from tqdm import trange
from tqdm.auto import tqdm
from algo.utils import plot_every_iteration
import scipy

def oicd(estimate_loss_fun,
         expectation_loss,
        #  fidelity, 
        ground_energy,
         n_shot, weights_dict, init_weights, num_iter,
         cyclic_mode=False,
         use_pratical_interp_flag=True,
         use_local_solvers_flag=False, 
         use_global_solvers_flag = False,
         use_eigen_method_flag = True,
         subproblem_method='BFGS',
         subproblem_iter=None,
         use_exact_update_frequencey_1_flag = False, # True only for all omegas = [1], [2]
         exact_mode=False, # for testing purpose, no noisy loss
         plot_flag=False,
         plot_argmin_flag = False,
         tol = 1e-2,
         refresh_print_bar = False 
         ):
    """
    Optimize VQA's weights using the Optimal Interpolation Coordinate Descent (OICD) method.

    Args:
        estimate_loss (function): A function to estimate the loss, taking weights and the number of samples as input and returning the estimated loss.
        expectation_loss (function): A function to compute the EXACT expectation loss, taking weights as input and returning the expectation loss.
        fidelity (function): A function to calculate fidelity, taking weights as input and returning the fidelity value.
        n_shot (int): The number of samples used for loss estimation.
        weights_dict (dict): A dictionary of weights, containing different weights with their corresponding omega values and interpolation nodes.
        init_weights (numpy.ndarray): The initial weights.
        num_iter (int): The number of iterations.
        cyclic_mode (bool, optional): Whether to enable cyclic mode, i.e., updating weights sequentially during each iteration. Defaults to False.
        use_pratical_interp_flag (bool, optional): Whether to use the practical OICD method (Algorithm 3 from the paper). Defaults to True.
        use_local_solvers_flag (bool, optional): Whether to use optimization solvers to solve the subproblems. Can be set to False when omega = [1]. Defaults to True.
        subproblem_method (str, optional): The method for solving the subproblems, such as 'CG' (Conjugate Gradient). Defaults to 'CG'.
        subproblem_iter (int, optional): The number of iterations for solving each subproblem. Defaults to 20.

    Returns:
        tuple: A tuple containing the optimized weights, the record of expectation loss values, and the record of fidelity values.
    """

    name = 'OICD'
    success = False

    def metric(weights):
        return np.abs(expectation_loss(weights)-ground_energy)

    if exact_mode: # for testing purpose
        estimate_loss = expectation_loss
    else:
        estimate_loss = lambda weights: estimate_loss_fun(weights, n_shot)

    weights = init_weights.copy()
    best_weights = init_weights.copy()

    true_loss = expectation_loss(weights)
    best_loss = true_loss
    fun_calling_count = 1
    # fid = fidelity(weights)
    # best_fid = fid
    approx_loss_value = true_loss

    expected_record_value = [true_loss]
    best_expected_record_value = [best_loss]   
    func_count_record_value= [fun_calling_count]
    # fidelity_record_value = [fid]
    # best_fid_record_value = [best_fid]   
    # approx_record_value = [approx_loss_value]
    metric_record = [metric(weights)]

    print("-"*100)

    m = len(weights)

    t = range(num_iter) if (refresh_print_bar is False) else trange(num_iter, desc="Bar desc", leave=True)
    for i in t:

        start = time.perf_counter()

        # choose a random index to update
        if cyclic_mode:
            j = i % m
        else:
            j = np.random.randint(m) if i == 0 else draw_new_j(m, prev_j)
            prev_j = j

        # read the info for interpolation
        omegas = weights_dict[f'weights_{j}']['omegas']
        interp_nodes = weights_dict[f'weights_{j}']["interp_nodes"]
        inv_A = weights_dict[f'weights_{j}']['inverse_interp_matrix']
        
        # execute interpolation
        if use_pratical_interp_flag:
            # Practical OICD Method in Algorithm 3 in paper
            shift = weights[j] - interp_nodes[0]
            shifted_interp_nodes = interp_nodes + shift
            E_s_inv = construct_Es_inv(shift, omegas)
            fun_vals = [approx_loss_value]
            weights_copy = weights.copy()
            for node in shifted_interp_nodes[1:]:
                weights_copy[j] = node
                fun_val = estimate_loss(weights_copy)
                fun_vals.append(fun_val)
            fun_vals = np.array(fun_vals)
            reco_coef = E_s_inv @ (inv_A @ fun_vals)
            fun_calling_count += 2*len(omegas) 
        else:
            #  Vanilla OICD Method in Algorithm 2 in paper
            fun_vals = []
            weights_copy = weights.copy()
            for node in interp_nodes: 
                weights_copy[j] = node
                fun_val = estimate_loss(weights_copy)
                fun_vals.append(fun_val)
            fun_vals = np.array(fun_vals)
            reco_coef = inv_A @ fun_vals
            fun_calling_count += (2*len(omegas)+1)
            
        # construct the approximate loss function
        r= len(omegas)

        def approx_loss(x):  
            trig_x_term = np.array([1 / np.sqrt(2)] + [func(omegas[k] * x).item() for k in range(r) for func in (np.cos, np.sin)])
            return np.dot(trig_x_term, reco_coef)
        
        # solve the subproblem: min approx_loss(x)
        if use_local_solvers_flag:

            def approx_loss_grad(x):
                trig_x_term = np.array([0] + [omegas[k] * func(omegas[k] * x).item() for k in range(r) for func in (lambda z: -np.sin(z), np.cos)])
                return np.dot(trig_x_term, reco_coef)

            def approx_loss_hess(x):
                trig_x_term = np.array([0] + [(omegas[k]**2) * func(omegas[k] * x).item() for k in range(r) for func in (lambda z: -np.cos(z), lambda z: -np.sin(z))])
                return np.dot(trig_x_term, reco_coef)
    
            # use optimization solvers for subproblem
            options = {'maxiter': subproblem_iter,
                       'disp': False}
            
            # IMPORTANT TIP: initial guess is set as the current coordinate value
            x0 = weights.copy()[j]

            # reference: 
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            
            # minimize can receive gradient and hessian           
            result = minimize(approx_loss, 
                              x0,
                              method = subproblem_method,
                              jac=approx_loss_grad,
                            #   hess=approx_loss_hess, # 'Newton-CG 或 'trust-constr
                              options=options)
            
            updated_weight = result.x.item() 
            approx_loss_value = result.fun 

        elif use_global_solvers_flag:
            
            # IMPORTANT TIP: initial guess is set as the current coordinate value
            x0 = weights.copy()[j]

            # Define bounds for the global optimizer
            # WARNING: this used the 2pi periodicity, only for equidistant frequency
            bounds = [(x0-np.pi, x0+np.pi)]

            # According to method, some need x0 or not, some need bounds or not.
            # reference: 
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            
            # Use global optimization solver
            result = sciopt.differential_evolution(approx_loss, 
                                                   bounds, 
                                                   strategy='best1bin', 
                                                   maxiter=subproblem_iter, 
                                                   disp=False)

            updated_weight = result.x.item()
            approx_loss_value = result.fun

        elif use_eigen_method_flag:
            
            x0 = weights.copy()[j]

            t_min, f_min = eigen_method_find_minimum(r, reco_coef)

            factor = weights_dict[f'weights_{j}']['scale_factor']
            updated_weight = t_min / factor
            approx_loss_value = f_min

        elif use_exact_update_frequencey_1_flag:
            
            factor = weights_dict[f'weights_{j}']['scale_factor']
            if len(omegas) != 1:
                raise ValueError(
                    f"Error: len(omegas) = {len(omegas)} is not equal to 1. `use_exact_update_frequencey_1_flag` cannot be True in this case."
                )

            x0 = weights.copy()[j]

            # we directly compute the analytical solution, 
            # only for case of omega = [1]
        
            a = reco_coef[0]/np.sqrt(2)
            b = reco_coef[1]
            c = reco_coef[2]
            updated_weight, approx_loss_value = update_for_frequency_1(a, b, c, approx_loss)
            updated_weight = updated_weight / factor
            
        if plot_argmin_flag and i % 10 == 0:

            # Plot the approx_loss function
            x_vals = np.linspace(x0 - np.pi, x0 + np.pi, 500)  # Range of x values for plotting
            y_vals = np.array([approx_loss(x) for x in x_vals])

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label="approx_loss(x)", color="blue")
            plt.scatter(x0, approx_loss(x0), color="green", label="Initial Point", zorder=5)
            plt.scatter(updated_weight, approx_loss(updated_weight), color="red", label="Optimal Point", zorder=5)
            plt.title("Approx Loss Function")
            plt.xlim(x0 - np.pi, x0 + np.pi)
            plt.xlabel("x")
            plt.ylabel("approx_loss(x)")
            plt.legend()
            plt.grid()
            plt.show()

            # Insert a delay of 1 second before showing the plot
            time.sleep(2)

    
        weights[j] = updated_weight

        # record the loss value
        true_loss = expectation_loss(weights)
        if true_loss < best_loss:
            best_loss = true_loss
            best_weights = weights.copy()

        # fid = fidelity(weights)
        # if fid > best_fid:
        #     best_fid = fid

        dist = metric(weights)
        metric_record.append(dist)

        expected_record_value.append(true_loss)
        best_expected_record_value.append(best_loss)
        func_count_record_value.append(fun_calling_count)
        # fidelity_record_value.append(fid)
        # best_fid_record_value.append(best_fid)
        # approx_record_value.append(approx_loss_value)
    

        end = time.perf_counter()
        elapsed = end - start
        message = f"Iter: {i}, {j}({m}), Metric: {dist:.4f}, Elapsed: {elapsed:.2f}s"

        if refresh_print_bar is True:
            t.set_description(f"[{name}] %s" % message)
            t.refresh()
        else:
            print(f"[{name}]", message)


        if plot_flag:
            plot_every_iteration(metric_record, name)    
            # plot_every_iteration(expected_record_value, fidelity_record_value, name)
            # plot_every_iteration(best_expected_record_value, fidelity_record_value, name)
            # plot_every_iteration(best_expected_record_value, best_fid_record_value, name)

        if np.abs(dist) < tol:
            success = True
            break
        
    return weights, success, best_expected_record_value, func_count_record_value, expected_record_value, metric_record

def draw_new_j(m, prev_j):
    j = np.random.randint(m)
    while prev_j is not None and j == prev_j:
        j = np.random.randint(m)
    return j

def update_for_frequency_1(a, b, c, approx_loss):
    """
    Update the weight for the case of omega = [1].
    the cost function has form: a + b*cos(x) + c*sin(x),
    its global minimizer is comptuted as follows.
    
    In general, the minimizer of the function a + b*cos(x) + c*sin(x) is
    given by the arctan(c/b).
    but when case of b=0 or c=0, the minimizer is any value.
    """

    # The goal here is to find the analytic solution of approx_loss, not exact_single_var_fun
    # And the solution should be within 0 to 2*pi

    if np.isclose(b, 0) and np.isclose(c, 0):
        # Constant function. Any value is an extrema, so it remains unchanged.
        updated_weight = 0.0
    elif np.isclose(b, 0) and not np.isclose(c, 0):
        # sin function, extrema influenced by amplitude c
        updated_weight = (3 * np.pi / 2) if c > 0 else (np.pi / 2)
    elif np.isclose(c, 0) and not np.isclose(b, 0):
        # cos function, extrema influenced by amplitude b
        updated_weight = np.pi if b > 0 else 0.0
    else: # not np.isclose(c, 0) and not np.isclose(b, 0)
        updated_weight = np.arctan(c / b)
        IS_MAXIMIZER = approx_loss(updated_weight) > a
        IS_POSITIVE = updated_weight > 0
        if IS_POSITIVE:
            if IS_MAXIMIZER:
                updated_weight += np.pi
        else:
            if IS_MAXIMIZER:
                updated_weight += np.pi
            else:
                updated_weight += 2 * np.pi
    
    approx_loss_value = approx_loss(updated_weight)

    return updated_weight, approx_loss_value



def construct_Es_inv(s, omegas):
    """
    Construct the matrix E_s^{-1} used for Algorithm 3: Practical OICD Method.
    See Lemma 1 in the paper.

    Example: when omega = [1], then
    E_s_inv = np.array([[1, 0, 0], 
                        [0, np.cos(omegas[0] * shift), - np.sin(omegas[0] * shift)], 
                        [0, np.sin(omegas[0] * shift), np.cos(omegas[0] * shift)]])
    """

    # Calculate the rotation matrices B_i^T
    num_blocks = len(omegas) + 1  # The first block is a 1x1 matrix [1], followed by 2x2 matrices for each Omega_i
    total_size = num_blocks * 2 - 1  # Compute the total size of the matrix
    
    # Initialize a zero matrix of size total_size x total_size
    E_s_inv = np.zeros((total_size, total_size))
    
    # Set the first block as the 1x1 matrix
    E_s_inv[0, 0] = 1
    
    # Fill in the subsequent rotation matrix blocks B_i^T
    for i, Omega_i in enumerate(omegas):
        # Construct each B_i^T
        B_i_T = np.array([[np.cos(Omega_i * s), -np.sin(Omega_i * s)],
                          [np.sin(Omega_i * s),  np.cos(Omega_i * s)]])
        # Place B_i^T in the matrix E_s_inv
        E_s_inv[2*i+1:2*i+3, 2*i+1:2*i+3] = B_i_T
    
    return E_s_inv



# def compute_real_roots_fourier_series(a0, a_coeffs, b_coeffs, tolerance=1e-8):
#     """
#     Computes the real roots of a finite Fourier series using the eigenvalue method.

#     Parameters:
#     - a0: Constant term coefficient.
#     - a_coeffs: List or array of cosine coefficients [a1, a2, ..., aN].
#     - b_coeffs: List or array of sine coefficients [b1, b2, ..., bN].
#     - tolerance: Threshold to consider the imaginary part negligible when computing roots (default is 1e-8).

#     Returns:
#     - roots: Array of real roots within the interval [0, 2π).
#     """
#     N = len(a_coeffs)
#     if len(b_coeffs) != N:
#         raise ValueError("The lengths of a_coeffs and b_coeffs must be equal.")

#     # Total number of terms in the Fourier series: 2N + 1
#     total_terms = 2 * N + 1

#     # Define the array h_k according to the given definition
#     h = np.zeros(total_terms, dtype=complex)

#     # For k = 0 to N-1
#     for k in range(N):
#         h[k] = a_coeffs[N - k - 1] + 1j * b_coeffs[N - k - 1]

#     # For k = N
#     h[N] = 2 * a0

#     # For k = N+1 to 2N
#     for k in range(N + 1, total_terms):
#         h[k] = a_coeffs[k - N - 1] - 1j * b_coeffs[k - N - 1]

#     # Construct the 2N x 2N matrix B
#     size = 2 * N
#     B = np.zeros((size, size), dtype=complex)

#     # Fill the upper diagonal with 1s (Kronecker delta δ_{j, k-1})
#     for j in range(size - 1):
#         B[j, j + 1] = 1.0  # B[j, k] where j = 0..2N-2, k = j+1

#     # Set the last row according to the given definition
#     denominator = a_coeffs[-1] - 1j * b_coeffs[-1]  # a_N - i * b_N
#     if np.islose(denominator, 1e-10):
#         raise ValueError("Denominator in the last row of matrix B is zero.")
#     # This case implies both a_N and b_N are zero, which is not allowed.

#     B[size - 1, :] = -h[:size] / denominator

#     # Compute the eigenvalues of matrix B
#     eigenvalues, _ = scipy.linalg.eig(B)

#     # (Optional) Visualize the eigenvalue distribution on the complex plane
#     # plot_eigenvalues_on_complex_plane(B)

#     # Compute the roots t_k = -i * log(z_k)
#     # Only retain roots with negligible imaginary parts
#     real_roots = []
#     for z in eigenvalues:
#         if np.isclose(z, 1e-8):
#             continue  # Avoid log(0)
#         log_z = np.log(z)
#         t_k = -1j * log_z
#         # Check if the imaginary part is negligible
#         if np.abs(t_k.imag) < tolerance:
#             # Normalize t_k to be within [0, 2π)
#             t_k_real = np.mod(t_k.real, 2 * np.pi)
#             real_roots.append(t_k_real)

#     # Convert to a NumPy array
#     real_roots = np.array(real_roots)

#     # Remove duplicate roots within the given tolerance
#     # real_roots = np.unique(np.round(real_roots, decimals=8))  # Adjust decimals as needed

#     # Sort the real roots in ascending order
#     real_roots = np.sort(real_roots)

#     return real_roots


def compute_real_roots_fourier_series(a0, a_coeffs, b_coeffs, tolerance=1e-8):
    """
    Computes the real roots of a finite Fourier series using the eigenvalue method.

    Parameters:
    - a0: Constant term coefficient.
    - a_coeffs: List or array of cosine coefficients [a1, a2, ..., aN].
    - b_coeffs: List or array of sine coefficients [b1, b2, ..., bN].
    - tolerance: Threshold to consider the imaginary part negligible when computing roots (default is 1e-8).

    Returns:
    - roots: Array of real roots within the interval [0, 2π).
    """
    N = len(a_coeffs)
    if len(b_coeffs) != N:
        raise ValueError("The lengths of a_coeffs and b_coeffs must be equal.")

    # Check the last elements of a_coeffs and b_coeffs
    while abs(a_coeffs[-1]) < tolerance and abs(b_coeffs[-1]) < tolerance:
        # If both the last coefficients are too small (near-zero), remove them
        a_coeffs = a_coeffs[:-1]
        b_coeffs = b_coeffs[:-1]
        N -= 1  # Decrease N, since we've removed one coefficient

        # Check if we have reduced all coefficients
        if N == 0:
            raise ValueError("All coefficients are too small, unable to compute roots.")

    # Total number of terms in the Fourier series: 2N + 1
    total_terms = 2 * N + 1

    # Define the array h_k according to the given definition
    h = np.zeros(total_terms, dtype=complex)

    # For k = 0 to N-1
    for k in range(N):
        h[k] = a_coeffs[N - k - 1] + 1j * b_coeffs[N - k - 1]

    # For k = N
    h[N] = 2 * a0

    # For k = N+1 to 2N
    for k in range(N + 1, total_terms):
        h[k] = a_coeffs[k - N - 1] - 1j * b_coeffs[k - N - 1]

    # Construct the 2N x 2N matrix B
    size = 2 * N
    B = np.zeros((size, size), dtype=complex)

    # Fill the upper diagonal with 1s (Kronecker delta δ_{j, k-1})
    for j in range(size - 1):
        B[j, j + 1] = 1.0  # B[j, k] where j = 0..2N-2, k = j+1

    # Set the last row according to the given definition
    denominator = a_coeffs[-1] - 1j * b_coeffs[-1]  # a_N - i * b_N
    if denominator == 0:
        raise ValueError("Denominator in the last row of matrix B is zero.")
    # This case implies both a_N and b_N are zero, which is not allowed.

    B[size - 1, :] = -h[:size] / denominator

    # Compute the eigenvalues of matrix B
    eigenvalues, _ = scipy.linalg.eig(B)

    # (Optional) Visualize the eigenvalue distribution on the complex plane
    # plot_eigenvalues_on_complex_plane(B)

    # Compute the roots t_k = -i * log(z_k)
    # Only retain roots with negligible imaginary parts
    real_roots = []
    for z in eigenvalues:
        if np.isclose(z, 1e-8):
            continue  # Avoid log(0)
        log_z = np.log(z)
        t_k = -1j * log_z
        # Check if the imaginary part is negligible
        if np.abs(t_k.imag) < tolerance:
            # Normalize t_k to be within [0, 2π)
            t_k_real = np.mod(t_k.real, 2 * np.pi)
            real_roots.append(t_k_real)

    # Convert to a NumPy array
    real_roots = np.array(real_roots)

    # Remove duplicate roots within the given tolerance
    # real_roots = np.unique(np.round(real_roots, decimals=8))  # Adjust decimals as needed

    # Sort the real roots in ascending order
    real_roots = np.sort(real_roots)

    return real_roots



def eigen_method_find_minimum(r, reco_coef):
    """
    Finds the minimum of a Fourier series using the eigenvalue method.

    Parameters:
    - r: The number of terms in the Fourier series (also determines the length of reco_coef).
    - reco_coef: The Fourier coefficients, where the first term is the constant (a0),
                 followed by alternating cosine (a) and sine (b) coefficients.

    Returns:
    - t_min: The value of t that minimizes the Fourier series.
    - f_min: The minimum value of the Fourier series at t_min.
    """
    
    N = r

    # First part: The constant term coefficient (a0)
    a0 = reco_coef[0] / np.sqrt(2)

    # Second part: Even-indexed coefficients (starting from index 1, step size 2)
    a_coeffs = reco_coef[1::2]

    # Third part: Odd-indexed coefficients (starting from index 2, step size 2)
    b_coeffs = reco_coef[2::2]
    
    # Compute the Fourier coefficients of the derivative of the series
    a0_deriv = 0.0  # The constant term derivative is zero
    a_coeffs_deriv = [b * n for n, b in enumerate(b_coeffs, start=1)]  # Derivatives of cosine terms
    b_coeffs_deriv = [-a * n for n, a in enumerate(a_coeffs, start=1)]  # Derivatives of sine terms
    
    # Find the real roots of f'(t) = 0 using the eigenvalue method
    try:
        roots_eigen_fprime = compute_real_roots_fourier_series(a0_deriv, a_coeffs_deriv, b_coeffs_deriv)
    except Exception as e:
        print(f"Error in eigenvalue method for f'(t): {e}")
        roots_eigen_fprime = []
    
    # Define the Fourier series function f(t)
    def f(t):
        result = a0
        for n in range(1, N+1):
            result += a_coeffs[n-1] * np.cos(n * t) + b_coeffs[n-1] * np.sin(n * t)
        return result
    
    # Calculate f(t) at each root of f'(t) = 0
    f_values = [f(root) for root in roots_eigen_fprime]
    
    # Find the minimum value of f(t) and the corresponding t
    if f_values:
        min_index = np.argmin(f_values)
        t_min = roots_eigen_fprime[min_index]
        f_min = f_values[min_index]
    else:
        t_min, f_min = None, None
    
    return t_min, f_min
