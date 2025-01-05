import numpy as np
from tqdm import trange  # tqdm is used for displaying progress bars.
import matplotlib.pyplot as plt
from IPython.display import clear_output
from algo.utils_qiskit import parameter_shift_for_equidistant_frequencies
from algo.utils_qiskit import plot_every_iteration

def rcd(estimate_loss_fun, 
        expectation_loss,
        fidelity,
        n_shot, weights_dict, init_weights, num_iter,
        cyclic_mode=False,
        learning_rate=0.1,
        decay_step=30, 
        decay_rate=-1.0, # -1.0 means no decay
        decay_threshold=1e-4, # 1e-4 is the minimum learning rate
        exact_mode=False,
        plot_flag=False,
        ):

    name = 'RCD'

    if exact_mode:
        estimate_loss = expectation_loss
    else:
        estimate_loss = lambda weights: estimate_loss_fun(weights, n_shot)

    weights = init_weights.copy()
    best_weights = init_weights.copy()

    true_loss = expectation_loss(weights)
    best_loss = true_loss
    fun_calling_count = 1
    fid = fidelity(weights)
    best_fid = fid

    expected_record_value = [true_loss]
    best_expected_record_value = [best_loss]   
    func_count_record_value= [fun_calling_count]
    fidelity_record_value = [fid]
    best_fid_record_value = [best_fid]   

    print("-"*100)

    t = trange(num_iter, desc="Bar desc", leave=True)
    m = len(weights)

    for i in t:

        # choose a random index to update
        if cyclic_mode:
            j = i % m
        else:
            j = np.random.randint(m)
        
        # read the info for parameter shift rule
        omegas = weights_dict[f'weights_{j}']['omegas']
        factor = weights_dict[f'weights_{j}']['scale_factor']
        gradient_j = parameter_shift_for_equidistant_frequencies(estimate_loss, weights, j, omegas, factor)
        fun_calling_count += 2*len(omegas)

        if decay_rate > 0 and (i + 1 ) % decay_step == 0:
            learning_rate = learning_rate * decay_rate
            learning_rate = max(learning_rate, decay_threshold)

        weights[j] = weights[j] - learning_rate * gradient_j

        # record the loss value
        true_loss = expectation_loss(weights)
        if true_loss < best_loss:
            best_loss = true_loss
            best_weights = weights.copy()

        fid = fidelity(weights)
        if fid > best_fid:
            best_fid = fid

        expected_record_value.append(true_loss)
        best_expected_record_value.append(best_loss)
        func_count_record_value.append(fun_calling_count)
        fidelity_record_value.append(fid)
        best_fid_record_value.append(best_fid)

        message = f"Iter: {i}, {j}({m}), Best loss: {best_loss:.4f}, Cur. loss: {true_loss:.4f}, Best Fid.: {best_fid:.4f}, Cur. Fid.: {fid:.4f}"
        # message = f"Iter: {i}, Best loss: {best_loss}, True loss: {true_loss}, Fidelity: {fid}"
        t.set_description(f"[{name}] %s" % message)
        t.refresh()

        if plot_flag and i % 10 == 0:
            plot_every_iteration(expected_record_value, fidelity_record_value, name)
            # plot_every_iteration(best_expected_record_value, fidelity_record_value, name)
            # plot_every_iteration(best_expected_record_value, best_fid_record_value, name)

        if np.abs(fid - 1) < 1e-3:
            break

    return best_weights, best_expected_record_value, best_fid_record_value, func_count_record_value, expected_record_value, fidelity_record_value
