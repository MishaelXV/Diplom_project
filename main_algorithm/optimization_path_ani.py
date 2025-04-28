import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import simpson
from optimizator.optimizer import compute_leakage_profile
from main_algorithm.constants import COMMON_CONSTANTS

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

def calculate_metric(z, found_left, found_right, true_left, true_right, Pe_opt):
    if len(found_left) <= 2:
        Pe_opt = [Pe[0], 0]

    step_opt = compute_leakage_profile(z, found_left, found_right, Pe_opt)
    step_true = compute_leakage_profile(z, true_left, true_right, Pe)

    squared_diff = np.abs(step_opt - step_true)

    area = simpson(squared_diff)

    L = max(z) - min(z)

    metric = np.sqrt(area / L)

    return metric


def plot_optimization_path(df_history, found_left, found_right, x_data):
    pe2_column = 'Pe_1'
    pe3_column = 'Pe_2'

    if pe2_column not in df_history.columns or pe3_column not in df_history.columns:
        print(f"Ошибка: Один или оба столбца {pe2_column} или {pe3_column} отсутствуют в DataFrame.")
        return None

    x_vals = np.linspace(0, Pe[0] + 500, 150)
    y_vals = np.linspace(0, Pe[0] + 500, 150)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Pe_opt = [Pe[0], X[i, j], Y[i, j], 0]
            Z[i, j] = calculate_metric(x_data, found_left, found_right,
                                                     true_boundaries['left'], true_boundaries['right'], Pe_opt)
    fig, ax = plt.subplots(figsize=(10, 8))
    cs = plt.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.7)
    plt.colorbar(cs, label="Невязка")

    plt.scatter([Pe[1]], [Pe[2]], c='green', s=200,
                             label="Истинные значения", edgecolor='black')

    plt.scatter([df_history[pe2_column].iloc[0]],
                              [df_history[pe3_column].iloc[0]],
                              c='blue', s=100, marker='o', label="Начальная точка")

    path_line, = plt.plot([], [], linestyle='-', linewidth=2, label="Траектория")
    current_point = plt.scatter([df_history[pe2_column].iloc[0]],
                                [df_history[pe3_column].iloc[0]],
                                c='red', s=100, marker='o', label="Текущая позиция")

    plt.xlabel('Pe_2')
    plt.ylabel('Pe_3')
    plt.title("Траектория поиска минимума")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)

    def init():
        path_line.set_data([], [])
        current_point.set_offsets(np.array([[df_history[pe2_column].iloc[0],
                                             df_history[pe3_column].iloc[0]]]))
        return path_line, current_point

    def update(frame):
        if frame >= len(df_history):
            frame = len(df_history) - 1

        x_path = df_history[pe2_column].values[:frame + 1]
        y_path = df_history[pe3_column].values[:frame + 1]

        Pe_max = Pe[0]
        Pe_min = 0

        x_path = np.clip(x_path, Pe_min, Pe_max)
        y_path = np.clip(y_path, Pe_min, Pe_max)

        path_line.set_data(x_path, y_path)
        current_point.set_offsets(np.array([[x_path[-1], y_path[-1]]]))
        return path_line, current_point

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df_history) + 3,
        init_func=init,
        blit=True,
        interval=300,
        repeat=True
    )

    plt.tight_layout()
    plt.show()
    return ani