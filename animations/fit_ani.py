from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from animations.constants import COMMON_CONSTANTS
from main_block.main_functions import main_func
from optimizator.optimizer import compute_leakage_profile

true_boundaries = COMMON_CONSTANTS['boundaries']
Pe = COMMON_CONSTANTS['Pe']
N = COMMON_CONSTANTS['N']
sigma = COMMON_CONSTANTS['sigma']
TG0 = COMMON_CONSTANTS['TG0']
atg = COMMON_CONSTANTS['atg']
A = COMMON_CONSTANTS['A']

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

def run_ani(x_data, y_data_noize, df_history, found_left, found_right):
    fig, (ax_leak, ax_temp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    fig.subplots_adjust(right=0.88)

    leak_line, = ax_leak.step([], [], where='post', color='blue', label='Найденный', linewidth=2.5)
    ax_leak.step(x_data, compute_leakage_profile(x_data, true_boundaries['left'], true_boundaries['right'], Pe),
                                   where='post', color='red', label='Истинный', linewidth=2.5)

    temp_line, = ax_temp.plot([], [], color='blue', label='Найденный', linewidth=2)
    ax_temp.plot(x_data, y_data_noize, color='red', label='Истинный', linewidth=2)

    ax_leak.legend(frameon=False)
    ax_leak.set_ylabel('ΔPe')
    ax_leak.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax_leak.set_ylim(0, Pe[0])

    ax_temp.legend(frameon=False)
    ax_temp.set_xlabel('z/rw')
    ax_temp.set_ylabel('θ')
    ax_temp.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    fig.text(0.91, 0.755, 'Профиль утечки', va='center', ha='left',
             rotation=-90, fontsize=16, fontweight='bold', color='black')
    fig.text(0.91, 0.37, 'Температурный профиль', va='center', ha='left',
             rotation=-90, fontsize=16, fontweight='bold', color='black')

    fig.suptitle('Оптимизация профилей', fontsize=18, fontweight='bold', y=0.95)

    def init():
        leak_line.set_data([], [])
        temp_line.set_data([], [])
        return leak_line, temp_line

    def update(frame):
        row = df_history.iloc[frame]
        Pe_columns = [col for col in df_history.columns if col.startswith('Pe_')]
        Pe_middle = row[Pe_columns].values.tolist()
        Pe_opt = [Pe[0]] + Pe_middle + [0]

        y_temp = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)
        leak_profile = compute_leakage_profile(x_data, found_left, found_right, Pe_opt)

        leak_line.set_data(x_data, leak_profile)
        temp_line.set_data(x_data, y_temp)

        return leak_line, temp_line
    ani = FuncAnimation(fig, update, frames=len(df_history), init_func=init, blit=True, repeat=True)
    plt.show()
    return ani