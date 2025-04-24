import matplotlib.pyplot as plt
from calculates_block.data import generate_data, noize_data, smooth_data
from calculates_block.main_functions import main_func
from optimizator.optimizer import run_optimization, compute_leakage_profile, calculate_deviation_metric, reconstruct_Pe_list
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300
})

#Константы
boundary_dict = {'left': [0, 150, 300, 450], 'right': [100, 250, 350, 500]}
Pe = [5000, 1000, 500,  0]
zInf = 100000
atg = 0.0001
TG0 = 1
A = 6
sigma = 0.001
N = 500


def main():
    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, N, sigma, A, model_ws, model_ms)

    result, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    Pe_opt = reconstruct_Pe_list(result.params, Pe[0])
    y_temp = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)

    metric = calculate_deviation_metric(result.params, x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe)
    print(metric)
    print(df_history)

    fig, (ax_leak, ax_temp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    fig.subplots_adjust(right=0.88)

    ax_leak.step(x_data, compute_leakage_profile(x_data, found_left, found_right, Pe_opt),
                 where='post', color='blue', label='Найденный', linewidth=2.5)
    ax_leak.step(x_data, compute_leakage_profile(x_data, boundary_dict['left'], boundary_dict['right'], Pe) , label = 'Истинный', color='red', linewidth=2.5)
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
    plt.savefig('Профиль утечки.png', format = 'png', dpi = 300)
    plt.savefig('Профиль утечки.pdf', format = 'pdf')
    plt.close()


if __name__ == "__main__":
    main()