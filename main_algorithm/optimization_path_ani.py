import numpy as np
from lmfit import Parameters
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from calculates_block.main_functions import main_func
from optimizator.optimizer import calculate_deviation_metric
from main_algorithm.constants import COMMON_CONSTANTS

def plot_optimization_path(df_history, Pe_true, found_left, found_right, x_data, y_data):
    if len(df_history.columns) < 3:
        print("Недостаточно параметров для построения контурного графика")
        return None

    pe_columns = [col for col in df_history.columns if col.startswith('Pe_')]
    if len(pe_columns) < 2:
        print("Нужно минимум 2 параметра Pe для построения контурного графика")
        return None

    x_vals = np.linspace(0, Pe_true[0] * 1.5, 50)
    y_vals = np.linspace(0, Pe_true[1] * 1.5, 50)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            test_pe = [X[i, j], Y[i, j]] + [0]
            params = Parameters()
            for idx in range(len(test_pe) - 1):
                params.add(f'Pe_{idx + 1}', value=test_pe[idx], min=0, max=30000)

            model_func = lambda x_: main_func(params, x_, 100000, COMMON_CONSTANTS['TGO'], COMMON_CONSTANTS['atg'],
                                              COMMON_CONSTANTS['A'], found_left, found_right)
            try:
                Z[i, j] = calculate_deviation_metric(model_func, x_data, y_data)
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = plt.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.7)
    plt.colorbar(cs, label="Отклонение модели")

    true_point = plt.scatter([Pe_true[0]], [Pe_true[1]], c='green', s=200,
                             label="Истинные значения", edgecolor='black')

    start_point = plt.scatter([df_history[pe_columns[0]].iloc[0]],
                              [df_history[pe_columns[1]].iloc[0]],
                              c='blue', s=100, marker='o', label="Начальная точка")

    path_line, = plt.plot([], [], linestyle='-', linewidth=2, label="Траектория")
    current_point = plt.scatter([df_history[pe_columns[0]].iloc[0]],
                                [df_history[pe_columns[1]].iloc[0]],
                                c='red', s=100, marker='o', label="Текущая позиция")

    plt.xlabel(pe_columns[0], fontsize=12)
    plt.ylabel(pe_columns[1], fontsize=12)
    plt.title("Траектория поиска минимума", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    def init():
        path_line.set_data([], [])
        current_point.set_offsets(np.array([[df_history[pe_columns[0]].iloc[0],
                                             df_history[pe_columns[1]].iloc[0]]]))
        return path_line, current_point

    def update(frame):
        if frame >= len(df_history):
            frame = len(df_history) - 1

        x_path = df_history[pe_columns[0]].values[:frame + 1]
        y_path = df_history[pe_columns[1]].values[:frame + 1]

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