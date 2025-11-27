import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from algo.utils import load
import os
from matplotlib.ticker import MaxNLocator
import matplotlib

# Update matplotlib settings with specified fonts and sizes
fontsize = 20
# 全局设置 dpi 和 bbox_inches
matplotlib.rcParams.update({
    'savefig.dpi': 400,  # 设置保存图像时的 DPI
    'savefig.bbox': 'tight',  # 设置保存时去除空白区域
    'font.family': 'serif',
    'axes.labelsize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': 12,
    'axes.titlesize': 20,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'lines.linewidth': 3,
    # 'text.usetex': True,
    'figure.figsize': (6, 5), ###
})

def plot_results(data_list, save_path, JOB_ID, phys, x_right=400):
    """
    Plot averaged performance (metric vs function calls) over multiple experiments.
    """
    colors = {
        'OICD': "#2aa1f6",
        'RCD':  '#ff7f0e',
        'GD':   '#2ca02c',
    }

    algos = colors.keys()

    os.makedirs(save_path, exist_ok=True)

    for algo in algos:
        xs_list, ys_list = [], []

        for data in data_list:
            try:
                xs_list.append(data[algo]['fun_calls'])
                ys_list.append(data[algo]['metrics'])
            except KeyError:
                print(f"[Warning] Missing data for {algo} in one run, skipped.")
                continue

        if not xs_list:
            continue

        max_len = max(len(x) for x in xs_list)
        padded_ys = [
            np.pad(y, (0, max_len - len(y)), constant_values=y[-1])
            for y in ys_list
        ]
        mean_y = np.mean(padded_ys, axis=0)
        max_x = max(xs_list, key=len)

        if algo == 'GD':
            plt.plot(max_x, mean_y, label='SGD', linewidth=3, color=colors[algo])
        elif algo == 'OICD':
            plt.plot(max_x, mean_y, label='ICD', linewidth=3, color=colors[algo])
        else:
            plt.plot(max_x, mean_y, label=algo, linewidth=3, color=colors[algo])

    plt.xlabel("Number of function evaluations")
    # plt.ylabel(r"Error $\left|\langle O\rangle_{\mathrm{ground}} - \langle O\rangle_{\theta}\right|$")
    plt.ylabel("Metric")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("N=12") ############
    plt.ylim(bottom=0)
    plt.xlim(left=-1, right=x_right)
    # plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    plot_file = os.path.join(save_path, f"{JOB_ID}_{phys}_performance.png")
    plt.savefig(plot_file, dpi=400)
    plt.close()  # Free up memory
    print(f"✅ Plot saved to {plot_file}")

if __name__ == "__main__":
    JOB_ID = 506041 ############
    # phys = "tfim_HVA_Wiersema" ############
    phys = "XXZ_HVA_Wiersema_new" ############
    path = f"{JOB_ID}/{phys}"
    data_path = os.path.join(path, 'data_dict.pkl')

    try:
        all_data = load(data_path)
    except Exception as e:
        print(f"❌ Failed to load data from {data_path}: {e}")
    else:
        # 400, 600, 800, 1600, 5000, 40000, 60000
        plot_results(all_data, path, JOB_ID, phys, x_right=60000) ############
