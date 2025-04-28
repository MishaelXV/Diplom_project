import matplotlib.pyplot as plt
from main_block.main_functions import main_func, reconstruct_Pe_list
from main_algorithm.constants import COMMON_CONSTANTS
from optimizator.optimizer import compute_leakage_profile

TG0 = COMMON_CONSTANTS['TG0']
atg = COMMON_CONSTANTS['atg']
A = COMMON_CONSTANTS['A']
Pe = COMMON_CONSTANTS['Pe']
true_boundaries = COMMON_CONSTANTS['boundaries']

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def plot_comparison(x_data, y_data_noize, params, found_left, found_right):
    Pe_opt = reconstruct_Pe_list(params, COMMON_CONSTANTS['Pe'][0])
    y_temp = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)

    fig, (ax_leak, ax_temp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    fig.subplots_adjust(right=0.88)

    ax_leak.step(x_data, compute_leakage_profile(x_data, found_left, found_right, Pe_opt),
                 where='post', color='blue', label='Найденный', linewidth=2.5)
    ax_leak.step(x_data, compute_leakage_profile(x_data, true_boundaries['left'], true_boundaries['right'], Pe),
                 where='post', label='Истинный', color='red', linewidth=2.5)
    ax_leak.legend(frameon=False)
    ax_leak.set_ylabel('ΔPe')
    ax_leak.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    ax_temp.plot(x_data, y_temp, label='Найденный', color='blue', linewidth=2)
    ax_temp.plot(x_data, y_data_noize, label='Истинный', color='red', linewidth=2)
    ax_temp.legend(frameon=False)
    ax_temp.set_xlabel('z/rw')
    ax_temp.set_ylabel('θ')
    ax_temp.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    fig.text(0.91, 0.755, 'Профиль утечки', va='center', ha='left',
             rotation=-90, fontsize=16, fontweight='bold', color='black')
    fig.text(0.91, 0.37, 'Температурный профиль', va='center', ha='left',
             rotation=-90, fontsize=16, fontweight='bold', color='black')

    fig.suptitle('Результаты моделирования и оптимизации', fontsize=18, fontweight='bold', y=0.95)
    plt.show()