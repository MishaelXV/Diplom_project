from lmfit import minimize, Parameters
import numpy as np
import pandas as pd
from calculates_block.utils import main_func

rng = np.random.default_rng(42)

def residuals_(params, x, y, zInf, TG0, atg, A, Pe, b, c):
    return main_func(params, x, zInf, TG0, atg, A, Pe, b, c) - y


def generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    x_data = np.linspace(b[0], c[-1], N)
    y_data = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, x_data, zInf, TG0, atg, A, Pe, b, c) + rng.normal(0, sigma, x_data.size)
    return x_data, y_data


def compute_relative_error(param_history, true_Pe):
    if not param_history or not true_Pe:
        print("Ошибка: пустые входные данные")
        return float('nan')

    last_params = param_history[-1][0]

    num_pe = len([k for k in last_params.keys() if k.startswith('Pe_')])
    if num_pe == 0:
        print("Ошибка: не найдены параметры Pe")
        return float('nan')

    predicted_Pe = np.array([last_params[f'Pe_{i + 1}'] for i in range(num_pe)])

    if len(true_Pe) < num_pe:
        print(f"Ошибка: true_Pe содержит {len(true_Pe)} элементов, но ожидается {num_pe}")
        return float('nan')

    true_Pe_array = np.array(true_Pe[:num_pe])

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs((predicted_Pe - true_Pe_array) / true_Pe_array) * 100
        relative_error[true_Pe_array == 0] = 0

    mean_relative_error = np.nanmean(relative_error)

    return mean_relative_error


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


def run_optimization(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    def residuals(params, x, y):
        return main_func(params, x, zInf, TG0, atg, A, Pe, b, c) - y

    x_data, y_data = generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N)
    params = create_parameters(Pe)

    param_history = []

    def iter_callback(params, iter, resid, *args, **kwargs):
        param_values = {param.name: float(param.value) for param in params.values()}
        chi_squared = float(np.sum(resid ** 2))
        param_history.append((param_values, chi_squared))

    result = minimize(residuals, params, args=(x_data, y_data), method='leastsq', iter_cb=iter_callback, ftol=1e-15)

    df_history = process_results(param_history)

    return result, param_history, df_history, x_data, y_data