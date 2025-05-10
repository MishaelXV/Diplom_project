import matplotlib.pyplot as plt
from main_block.data import generate_data, noize_data
from main_block.main_functions import main_func
from optimizator.bayes_optimizer import run_bayes_optimization
from optimizator.optimizer import run_optimization, compute_leakage_profile, calculate_deviation_metric
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

# Константы
boundary_dict = {"left": [0, 150, 300, 400],
                 "right": [100, 250, 350, 470]}
Pe = [20000, 10000, 5000, 0]
atg = 0.0001
TG0 = 1
A = 1
sigma = 0.01
N = 250

def plot_results(x_data, y_temp, y_data_noize, found_left, found_right, Pe_opt, Pe_true, output_path_png,
                 output_path_pdf):
    fig, (ax_leak, ax_temp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    fig.subplots_adjust(right=0.88)

    ax_leak.step(x_data, compute_leakage_profile(x_data, found_left, found_right, Pe_opt),
                 where='post', color='blue', label='Найденный', linewidth=2.5)
    ax_leak.step(x_data, compute_leakage_profile(x_data, boundary_dict['left'], boundary_dict['right'], Pe_true),
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

    plt.savefig(output_path_png, format='png', dpi=300)
    plt.savefig(output_path_pdf, format='pdf')
    plt.close()


def main():
    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    # found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, N, sigma, A, model_ws, model_ms)
    found_left, found_right = [0, 150, 300, 400], [100, 250, 350, 470]

    if len(found_left) == 1:
        Pe_opt = [0]

    elif len(found_left) == 2:
        Pe_opt = [Pe[0], 0]

    else:
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                              boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)
        # Pe_opt, df_history = run_bayes_optimization(x_data, y_data_noize, found_left, found_right,
        #                    boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, n_trials=200)

    y_temp = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)

    metric = calculate_deviation_metric(x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)
    plot_results(
        x_data, y_temp, y_data_noize,
        found_left, found_right,
        Pe_opt, Pe,
        output_path_png='Профиль утечки.png',
        output_path_pdf='Профиль утечки.pdf'
    )

    print(df_history)
    print(metric)


if __name__ == "__main__":
    main()