import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lmfit import minimize, Parameters
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from calculates_block.data import generate_data_optim
from calculates_block.main_functions import main_func
from regression.find_intervals import get_boundaries, load_training_data
from regression.optuna_search import train_models

def calculate_deviation_metric(model_func, x_data, y_data):
    x_dense = np.linspace(min(x_data), max(x_data), 300)

    data_interp = interp1d(x_data, y_data, kind='cubic', fill_value='extrapolate')
    y_data_dense = data_interp(x_dense)

    y_model_dense = model_func(x_dense)

    deviations = np.abs(y_model_dense - y_data_dense)

    total_area = simpson(deviations)

    L = max(x_data) - min(x_data)
    return total_area/L


def process_results(param_history):
    df_history = pd.DataFrame(param_history, columns=['parameters', 'Невязка'])
    df_params = pd.json_normalize(df_history['parameters'])
    df_history = pd.concat([df_history, df_params], axis=1)
    df_history = df_history.drop('parameters', axis=1)
    return df_history


def create_parameters(Pe):
    params = Parameters()
    for i in range(len(Pe) - 1):
        params.add(f'Pe_{i + 1}', value=1, min=0, max=30000)
    return params


def optimization_residuals(params, x, y, zInf, TG0, atg, A, Pe, left_boundary, right_boundary):
    return main_func(params, x, zInf, TG0, atg, A, Pe, left_boundary, right_boundary) - y


def run_ani(left_boundary, right_boundary, Pe, zInf, TG0, atg, A, sigma, N):
    x_data, y_data = generate_data_optim(left_boundary, right_boundary, Pe, zInf, TG0, atg, A, sigma, N)
    results = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms)
    found_left_boundary, found_right_boundary, z_norm, T_true_norm, T_noisy_norm, T_smooth, starts, ends = results

    params = create_parameters(Pe)
    param_history = []

    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()

    ax.plot(x_data, y_data, label='Профиль температуры', linewidth=2)
    line, = ax.plot([], [], 'g-', label='Модельная кривая', linewidth=2.5, alpha=0.8)

    fill = ax.fill_between([], [], [], color='red', alpha=0.3, label='Отклонение')

    ax.set_title('Подгонка кривой', fontsize=14, pad=15)
    ax.set_xlabel('z/rw', fontsize=12)
    ax.set_ylabel('θ', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    text_info = ax.text(0.84, 0.2, '', transform=ax.transAxes,
                        fontsize=11, bbox=dict(facecolor='white', alpha=0.85,
                                               edgecolor='gray', boxstyle='round,pad=0.5'),
                        verticalalignment='top')

    def init():
        line.set_data([], [])
        for coll in ax.collections[1:]:
            coll.remove()
        text_info.set_text('')
        return line, fill, text_info


    def update(frame):
        params_current, chi_sq, iter_num = frame

        temp_params = Parameters()
        for name, value in params_current.items():
            temp_params.add(name, value=value)

        current_Pe = [params_current[f"Pe_{i + 1}"] for i in range(len(left_boundary) - 1)]
        current_Pe.append(0)

        y_fit = main_func(temp_params, x_data, zInf, TG0, atg, A, current_Pe, found_left_boundary, found_right_boundary)

        line.set_data(x_data, y_fit)

        for coll in ax.collections[1:]:
            coll.remove()
        fill = ax.fill_between(x_data, y_data, y_fit, color='red', alpha=0.3)

        ax.set_title(f'Подгонка кривой (Итерация {iter_num})')

        info_text = f"Итерация: {iter_num}\nНевязка: {chi_sq:.2e}\n"
        for name, value in params_current.items():
            info_text += f"{name}: {value:.2f}\n"

        text_info.set_text(info_text)

        return line, fill, text_info

    def optimization_callback(params, iter, resid, param_history):
        param_values = {param.name: float(param.value) for param in params.values()}

        current_Pe = [param_values[f"Pe_{i + 1}"] for i in range(len(param_values))]
        current_Pe.append(0)

        model_func = lambda x_: main_func(params, x_, zInf, TG0, atg, A, current_Pe, found_left_boundary,
                                          found_right_boundary)

        deviation_metric = calculate_deviation_metric(model_func, x_data, y_data)

        param_history.append((param_values, deviation_metric, iter))

    result = minimize(
        lambda params, x, y: optimization_residuals(params, x, y, zInf, TG0, atg, A, Pe, found_left_boundary,
                                                    found_right_boundary),
        params,
        args=(x_data, y_data),
        method='leastsq',
        iter_cb=lambda params, iter, resid, *args, **kwargs:
        optimization_callback(params, iter, resid, param_history),
        ftol=1e-15,
        max_nfev=1000
    )

    ani = FuncAnimation(
        fig,
        update,
        frames=param_history[::1],
        init_func=init,
        blit=True,
        interval=400,
        repeat_delay=3000
    )

    plt.tight_layout()
    plt.show()

    df_history = process_results([(x[0], x[1]) for x in param_history])
    return result, param_history, df_history, x_data, y_data, ani


def plot_comparison(x_data, y_data, model_func):
    plt.figure(figsize=(10, 6))

    plt.plot(x_data, y_data, 'b-', label='Профиль температуры', linewidth=2)

    y_model = model_func(x_data)
    plt.plot(x_data, y_model, 'g-', label='Модельный профиль', linewidth=2.5, alpha=0.8)

    plt.fill_between(x_data, y_data, y_model,
                     color='red', alpha=0.3, label='Отклонение')

    title = 'Сравнение профилей'
    plt.title(title, fontsize=14)

    plt.xlabel('z/rw', fontsize=12)
    plt.ylabel('θ', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def optimized_model(x):
    current_Pe = [result.params[f"Pe_{i + 1}"].value for i in range(len(found_left) - 1)]
    current_Pe.append(0)
    return main_func(result.params, x, zInf, TG0, atg, A, current_Pe, found_left, found_right)


def plot_animated_residuals(df_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(-0.5, len(df_history) + 0.5)
    y_min = df_history['Невязка'].min() * 0.8
    y_max = df_history['Невязка'].max() * 1.2
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')

    ax.set_title('Изменение невязки по итерациям', fontsize=14)
    ax.set_xlabel('Номер итерации', fontsize=12)
    ax.set_ylabel('Значение невязки', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    line, = ax.plot([], [], 'b-', linewidth=2, label='Невязка')

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.8))

    def init():
        line.set_data([], [])
        text.set_text('')
        return line, text


    def update(frame):
        x = np.arange(frame + 1)
        y = df_history['Невязка'].values[:frame + 1]

        line.set_data(x, y)

        return line, text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df_history),
        init_func=init,
        blit=True,
        interval=200,
        repeat_delay=1000
    )

    plt.tight_layout()
    plt.show()
    return ani


def plot_animated_Pe(df_history, Pe_true):
    fig, ax = plt.subplots(figsize=(10, 6))

    pe_columns = [col for col in df_history.columns if col.startswith('Pe_')]

    ax.set_xlim(-0.5, len(df_history) + 0.5)
    y_min = df_history[pe_columns].min().min() * 0.8
    y_max = df_history[pe_columns].max().max() * 1.2
    ax.set_ylim(y_min, y_max)


    ax.set_title('Изменение Pe по итерациям', fontsize=14)
    ax.set_xlabel('Номер итерации', fontsize=12)
    ax.set_ylabel('Значение Pe', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    lines = []
    for i, col in enumerate(pe_columns):
        line, = ax.plot([], [], '-', linewidth=2, label=col)
        lines.append(line)

    true_lines = []

    for i, (col, true_val) in enumerate(zip(pe_columns, Pe_true[:-1])):
        hline = ax.axhline(true_val, linestyle='--', alpha=0.7,
                           label=f'Истинное {col} = {true_val}')
        true_lines.append(hline)

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(fontsize=10, loc='upper right')

    def init():
        for line in lines:
            line.set_data([], [])
        text.set_text('')
        return lines + [text]

    def update(frame):
        x = np.arange(frame + 1)

        for i, col in enumerate(pe_columns):
            y = df_history[col].values[:frame + 1]
            lines[i].set_data(x, y)

        return lines + [text]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df_history),
        init_func=init,
        blit=True,
        interval=200,
        repeat_delay=1000
    )

    plt.tight_layout()
    plt.show()
    return ani


def plot_optimization_path(df_history, Pe_true):
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

            model_func = lambda x_: main_func(params, x_, zInf, TG0, atg, A, test_pe, found_left, found_right)
            try:
                Z[i, j] = calculate_deviation_metric(model_func, x, y)
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = plt.contourf(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.7)
    plt.colorbar(cs, label="Отклонение модели")

    true_point = plt.scatter([Pe_true[0]], [Pe_true[1]], c='green', s=200,
                             marker='*', label="Истинные значения", edgecolor='black')

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


if __name__ == "__main__":
    boundary_dict = {'left': [0, 150, 300], 'right': [100, 250, 400]}
    Pe = [10000, 1000, 0]
    zInf = 100000
    TG0 = 1
    atg = 0.0001
    A = 5
    sigma = 0.0
    N = 500
    df = load_training_data()
    model_ws, model_ms = train_models(df)

    results = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms)
    found_left, found_right, z_norm, T_true_norm, T_noisy_norm, T_smooth, starts, ends = results

    result, history, df, x, y, anim = run_ani(boundary_dict['left'], boundary_dict['right'], Pe, zInf, TG0, atg, A, sigma, N)

    metric_value = calculate_deviation_metric(optimized_model, x, y)

    ani_optimization_path = plot_optimization_path(df, Pe)
    ani_residuals = plot_animated_residuals(df)
    ani_pe = plot_animated_Pe(df, Pe)

    plot_comparison(x, y, optimized_model)

