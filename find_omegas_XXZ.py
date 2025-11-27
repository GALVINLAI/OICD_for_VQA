import os
import pickle
import numpy as np

from algo.utils import (
    xxz_hamiltonian,
    circuit_QAOA_XXZ,
    expectation_loss_grad,
    interp_matrix, optimal_interp_points,
    check_is_trigometric,
    filter_omegas, is_equidistance_sequence,
)

def extract_and_save_compact_omegas(
    num_q: int,
    layer: int,
    scale_factor: float = 2.0,
    verbose: bool = False,
    save_dir: str = "results"
):
    """
    提取并保存某一 num_q 和 layer 情况下所有参数的紧致频率（compact omegas）
    保存内容：
    - .pkl 文件：包含每个参数对应的 omegas（结构化便于调用）
    """
    num_p = 4 * layer
    xxz_op = xxz_hamiltonian(num_q, Jz=0.5, bc='periodic')

    def expectation_loss(weights):
        return expectation_loss_grad(
            num_q, layer, weights, circuit=circuit_QAOA_XXZ, obs=xxz_op
        )

    omegas_1 = list(range(1, num_q // 2 + 1))
    omegas_2 = list(range(1, (num_q // 2) * 2 + 1))
    weights_dict = {}

    for j in range(num_p):
        omegas = omegas_1.copy() if j % 2 == 0 else omegas_2.copy()
        
        if verbose:
            print(f"\n[num_q={num_q}, layer={layer}, param={j}] initial omegas = {omegas}")


        for _ in range(5):
            weights = np.random.uniform(0, np.pi**2, num_p)
            coeffs, num_failed = check_is_trigometric(expectation_loss, j, omegas, weights, opt_interp_flag=True,
                                                    verbose=False)
            if num_failed is False:
                raise RuntimeError(f"Compact omegas invalid at param {j}: {omegas}")
            
            omegas = filter_omegas(coeffs, omegas, threshold=1e-7).tolist()

        if verbose:
            print(f"Compact omegas = {omegas}")

        # opt_mse, interp_nodes, inverse_interp_matrix = optimal_interp_points(compact)
        weights_dict[f'weights_{j}'] = {
            'omegas': omegas,  # 🔧 更新为 compact 版本
            'scale_factor': scale_factor,
            'interp_nodes': None,
            'inverse_interp_matrix': None,
            'opt_mse': None,
            'is_equidistance': is_equidistance_sequence(omegas),
        }

    os.makedirs(save_dir, exist_ok=True)

    # 保存 .pkl 文件（结构化字典）
    pkl_path = os.path.join(save_dir, f"compact_omegas_q{num_q}_l{layer}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(weights_dict, f)

    return weights_dict

# ========= 主程序 =========
if __name__ == "__main__":
    from prettytable import PrettyTable

    output_txt_path = os.path.join("find_omegas_XXZ_results", "all_tables_summary.txt")
    os.makedirs("find_omegas_XXZ_results", exist_ok=True)

    verbose_to_console = True  # ✅ 控制是否输出到屏幕

    with open(output_txt_path, "w", encoding="utf-8") as f:

        for num_q in range(18, 21, 2):
            layer = 2 * num_q
            weights_dict = extract_and_save_compact_omegas(
                num_q=num_q,
                layer=layer,
                verbose=True,
                save_dir="find_omegas_XXZ_results"
            )

            header = f"\n==== num_q={num_q}, layer={layer} ====\n"
            table = PrettyTable()
            table.field_names = ["Param ID", "Compact Omegas", "Is Equidist."]

            for j in range(4 * layer):
                omegas = weights_dict[f'weights_{j}']['omegas']
                is_eq = weights_dict[f'weights_{j}']['is_equidistance']
                table.add_row([j, omegas, "" if is_eq else False])

            # 写入文件
            print(header, file=f)
            print(table, file=f)

            # 可选是否输出到屏幕
            if verbose_to_console:
                print(header)
                print(table)


# import pickle

# with open("find_omegas_XXZ_results/compact_omegas_q5_l2.pkl", "rb") as f:
#     weights_dict = pickle.load(f)