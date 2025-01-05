import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shutil
import os
import numpy as np
import argparse
from glob import glob
from algo.utils_qiskit import load
import pickle

fontsize = 20
nice_fonts = {
    "font.family": "serif",
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": 15,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
}
matplotlib.rcParams.update(nice_fonts)
# colors = ['r', 'b', 'k', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phys', type=str, default='QAOA_XXZ_Wiersema', 
                        help='The physics system to be studied')
    parser.add_argument('--num_q', type=int, default=4, 
                        help='System size')
    parser.add_argument('--layer', type=int, default=2, 
                        help='The physics system to be studied')
    parser.add_argument('--Delta', type=float, default=0.5, 
                         help='The dimension of the problem')
    parser.add_argument('--repeat', type=int, default=3, 
                        help='The number of times to repeat the experiment')
    parser.add_argument('--x_lim_coff', type=float, default=6, 
                        help='The x-axis limit for the plots')
    parser.add_argument('--y_lim', type=float, default=0, 
                        help='The y-axis limit for the plots')
    parser.add_argument('--lr_gd', type=float, default=0.01, 
                        help='')
    parser.add_argument('--lr_rcd', type=float, default=0.01, 
                        help='')
    return parser.parse_args()

# Utility functions
def load_data(file_path):
    """Load data from a given file path."""
    data = load(file_path)  # Assuming 'load' is a predefined function
    for key in data.keys():
        data[key] = np.real(np.array(data[key]))
    return data

def calculate_statistics(all_data):
    """Calculate mean and std for loss data."""
    MEANS = {key: np.mean(values, axis=0) for key, values in all_data.items()}
    STDS = {key: np.std(values, axis=0) for key, values in all_data.items()}
    MAX = {key: np.max(values, axis=0) for key, values in all_data.items()}
    MIN = {key: np.min(values, axis=0) for key, values in all_data.items()}
    return MEANS, STDS, MAX, MIN

def plot_loss(function_calls_mean, MEANS, STDS, plot_path, fig_prefix, plot_xlim, args, showwhat='energy'):
    """Plot the loss data with mean and std."""
    plt.figure()
    for key, color in zip(MEANS.keys(), colors[:len(MEANS)]):
        plt.plot(function_calls_mean[key], MEANS[key], linewidth=3, color=color, label=key.upper())
        lower = MEANS[key] - STDS[key]
        upper = MEANS[key] + STDS[key]
        upper[upper > 1] = 1
        plt.fill_between(
            function_calls_mean[key], 
            lower, 
            upper, 
            color=color, alpha=0.2
        )

    plt.xlabel('Number of Function Evaluations')
    if showwhat == 'best_expected':
        plt.ylabel('Best Energy ratio: $E / E_{ground}$')
    elif showwhat == 'best_fid':
        plt.ylabel('Best Fidelity')
    elif showwhat == 'every_expected':
        plt.ylabel('Energy ratio: $E / E_{ground}$')
    elif showwhat == 'every_fid':
        plt.ylabel('Fidelity')

    plt.legend(fontsize=18, loc='lower right')
    plt.xlim(0, plot_xlim * args.x_lim_coff if args.x_lim_coff > 0 else plot_xlim)
    plt.ylim(args.y_lim, 1.05)
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, fig_prefix + showwhat + '_fun_evals.png'))
    plt.clf()

def pad_to_length(arr, target_length, padding_value=np.nan):
    """Pads the array to the target length with the padding value."""
    if len(arr) < target_length:
        padding = np.full(target_length - len(arr), padding_value)
        return np.concatenate([arr, padding])
    return arr

def pad_to_length_func(arr, target_length, algo, weights_dict):
    """Pads the array to the target length starting from the last element, with different increments based on `algo`."""

    padding_length = target_length - len(arr)
    padding = np.array([])

    num_p = len(weights_dict)
    one_gd = sum( 2*len(weights_dict[f'weights_{j}']['omegas']) for j in range(num_p))

    for _ in range(padding_length):
        
        if algo == "gd":
            d = one_gd
        elif algo == "rcd":
            d = 2*len(weights_dict[f'weights_{np.random.choice(num_p)}']['omegas'])
        elif algo == "oicd":
            d = 2*len(weights_dict[f'weights_{np.random.choice(num_p)}']['omegas'])
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        
        if len(padding) == 0:
            padding = np.array([arr[-1] + d])
        else:
            padding = np.append(padding, padding[-1] + d)
    
    return np.concatenate([arr, padding])

   

def main():
    args = parse_arguments()

    # Define paths
    exp_path = f'exp/{args.phys}'
    plot_path = f'plots/{args.phys}'

    fig_prefix = f'N{args.num_q}_layer{args.layer}_Delta{args.Delta}_repeat{args.repeat}_lr_rcd{args.lr_rcd}_lr_gd{args.lr_gd}_'

    # Check for and delete all files starting with fig_prefix in plot_path
    files_to_remove = glob(os.path.join(plot_path, f'{fig_prefix}*'))

    for file in files_to_remove:
        os.remove(file)
        print(f"Deleted file: {file}")

    # Ensure the plot directory exists (only creates if doesn't exist)
    os.makedirs(plot_path, exist_ok=True)

    # Get list of experiments
    num_exp = len(glob(f'{exp_path}/exp_*'))
    experiments = [f'exp_{i}' for i in range(num_exp)]

    algos = ['gd', 'rcd', 'oicd']

    ################################ prepare function calling data
    function_calls = {key: [] for key in algos}

    # Load and process data
    for experiment in experiments:
        file_path = os.path.join(exp_path, experiment, 'data_dict.pkl')
        data = load_data(file_path)

        for key in algos:
            function_calls[key].append(data[f'func_{key}'])

    # Pad the lists to have the same length
    max_len = max(
        max(len(lst) for lst in function_calls['gd']), 
        max(len(lst) for lst in function_calls['rcd']), 
        max(len(lst) for lst in function_calls['oicd'])
        )
    
    file_path = os.path.join(exp_path, 'exp_0', 'weights_dict.pkl')
    with open(file_path, 'rb') as file:
        weights_dict = pickle.load(file)

    # Pad loss lists and function calls to the same length
    for key in algos:
        function_calls[key] = [pad_to_length_func(arr, max_len, key, weights_dict) for arr in function_calls[key]]

    function_calls_mean, _, _, _ = calculate_statistics(function_calls)

    plot_xlim = max(len(lst) for lst in function_calls['oicd'])

    ################################
    # Initialize data structures
    best_expected_all = {key: [] for key in algos}
    every_expected_all = {key: [] for key in algos}
    best_fid_all = {key: [] for key in algos}
    every_fid_all = {key: [] for key in algos}

    # Load and process data
    for experiment in experiments:
        file_path = os.path.join(exp_path, experiment, 'data_dict.pkl')
        data = load_data(file_path)

        for key in algos:
            best_expected_all[key].append(data[f'best_expected_{key}'])
            every_expected_all[key].append(data[f'every_expected_{key}'])
            best_fid_all[key].append(data[f'fid_{key}'])
            every_fid_all[key].append(data[f'every_fid_{key}'])
 
    # Pad loss lists and function calls to the same length
    for key in algos:
        best_expected_all[key] = [pad_to_length(arr, max_len, 1) for arr in best_expected_all[key]]
        every_expected_all[key] = [pad_to_length(arr, max_len, 1) for arr in every_expected_all[key]]
        best_fid_all[key] = [pad_to_length(arr, max_len, 1) for arr in best_fid_all[key]]
        every_fid_all[key] = [pad_to_length(arr, max_len, 1) for arr in every_fid_all[key]]
    
    def plot_values(all, name):
        all = {key: np.vstack(values) for key, values in all.items()}
        MEAN, STD, MAX, MIN = calculate_statistics(all)
        plot_loss(function_calls_mean, MEAN, STD, plot_path, fig_prefix, plot_xlim, args, name)

    # plot_values(best_expected_all, 'best_expected')
    plot_values(every_expected_all, 'every_expected')
    # plot_values(best_fid_all, 'best_fid')
    plot_values(every_fid_all, 'every_fid')

if __name__ == "__main__":
    main()