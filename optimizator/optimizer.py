import numpy as np
import pandas as pd
from scipy.integrate import simpson
from lmfit import minimize, Parameters
from calculates_block.main_functions import main_func, reconstruct_Pe_list

def process_results(param_history):
    df = pd.DataFrame(param_history, columns=['parameters', 'Невязка', 'Итерация'])
    params_df = pd.json_normalize(df['parameters'])

    if 'Pe' in params_df:
        pe_list = params_df.pop('Pe')
        pe_df = pd.DataFrame(
            [pe[1:-1] for pe in pe_list],
            columns=[f"Pe_{i}" for i in range(1, len(pe_list[0]) - 1)]
        )
        params_df = pd.concat([params_df, pe_df], axis=1)

    return pd.concat([df.drop(columns='parameters'), params_df], axis=1)


def create_parameters(boundary, known_pe1):
    params = Parameters()
    n_layers = len(boundary) - 1

    for i in range(n_layers - 1):
        params.add(f"delta_{i}", value=0, min=0, max=known_pe1)

    sum_prev_deltas = " + ".join([f"delta_{i}" for i in range(n_layers - 1)])
    params.add(f"delta_{n_layers - 1}", expr=f"{known_pe1} - ({sum_prev_deltas})")

    return params


def calculate_deviation_metric(params, z, found_left, found_right, true_left, true_right, Pe_true):
    if len(found_left) <= 2:
        Pe_opt = [Pe_true[0], 0]
    else:
        Pe_opt = reconstruct_Pe_list(params, Pe_true[0])

    step_opt = compute_leakage_profile(z, found_left, found_right, Pe_opt)
    step_true = compute_leakage_profile(z, true_left, true_right, Pe_true)

    norm_factor = np.max(np.abs(step_true))
    norm_factor = norm_factor if norm_factor != 0 else 1.0

    squared_diff = ((step_opt - step_true) / norm_factor) ** 2

    area = simpson(squared_diff)

    L = max(z) - min(z)

    metric = np.sqrt(area / L)

    return metric


def optimization_residuals(params, x, y, TG0, atg, A, Pe, left_boundaries, right_boundaries):
    Pe = reconstruct_Pe_list(params, Pe[0])
    return main_func(x, TG0, atg, A, Pe, left_boundaries, right_boundaries) - y


def compute_leakage_profile(z, left_boundary, right_boundary, Pe_list):
    z = np.array(z)
    result = np.zeros_like(z)
    for i in range(len(right_boundary) - 1):
        mask = (z >= right_boundary[i]) & (z < left_boundary[i + 1])
        result[mask] = Pe_list[i] - Pe_list[i + 1]
    return result


def optimization_callback(params, iter, resid, param_history, z, found_left, found_right, true_left, true_right, Pe_true):
    param_values = {param.name: float(param.value) for param in params.values() if param.vary or 'delta' in param.name}
    deviation_metric = calculate_deviation_metric(params, z, found_left, found_right, true_left, true_right, Pe_true)
    Pe_current = reconstruct_Pe_list(params, Pe_true[0])
    param_values['Pe'] = Pe_current
    param_history.append((param_values, deviation_metric, iter))


def run_optimization(x_data, y_data, found_left, found_right, true_left, true_right, Pe, TG0, atg, A):
    known_pe1 = Pe[0]
    params = create_parameters(found_left, known_pe1)
    param_history = []

    result = minimize(
        lambda params, x, y: optimization_residuals(params, x, y, TG0, atg, A, Pe, found_left, found_right),
        params,
        args=(x_data, y_data),
        method='leastsq',
        iter_cb=lambda params, iter, resid, *args, **kwargs: optimization_callback(
            params, iter, resid, param_history, x_data, found_left, found_right, true_left, true_right, Pe),
        nan_policy='omit',
    )

    df_history = process_results(param_history)
    return result, df_history




