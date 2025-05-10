import numpy as np
from scipy.integrate import simpson
from lmfit import minimize, Parameters
from main_block.main_functions import main_func, reconstruct_Pe_list
from optimizator.process import process_results

def create_parameters(boundary, known_pe1):
    params = Parameters()
    n_layers = len(boundary) - 1

    for i in range(n_layers - 1):
        params.add(f"delta_{i}", value=0, min=0, max=known_pe1)

    sum_prev_deltas = " + ".join([f"delta_{i}" for i in range(n_layers - 1)])
    params.add(f"delta_{n_layers - 1}", expr=f"{known_pe1} - ({sum_prev_deltas})")

    return params


def calculate_deviation_metric(z, found_left, found_right, true_left, true_right, Pe_true, Pe_opt):
    step_opt = compute_leakage_profile(z, found_left, found_right, Pe_opt)
    step_true = compute_leakage_profile(z, true_left, true_right, Pe_true)

    diff_abs = np.abs(step_opt - step_true)
    true_abs = np.abs(step_true)
    l1_diff = simpson(diff_abs, x=z)
    l1_true = simpson(true_abs, x=z)
    relative_l1_error_percent = (l1_diff / l1_true) * 100

    return relative_l1_error_percent


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
    Pe_current = reconstruct_Pe_list(params, Pe_true[0])
    deviation_metric = calculate_deviation_metric(z, found_left, found_right, true_left, true_right, Pe_true, Pe_current)

    param_values['Pe'] = Pe_current
    param_values['residuals'] = float(np.sum(resid**2))

    param_history.append((param_values, deviation_metric, iter))


def run_optimization(x_data, y_data, found_left, found_right, true_left, true_right, Pe, TG0, atg, A, method = 'leastsq'):
    known_pe1 = Pe[0]
    params = create_parameters(found_left, known_pe1)
    param_history = []

    result = minimize(
        lambda params, x, y: optimization_residuals(params, x, y, TG0, atg, A, Pe, found_left, found_right),
        params,
        args=(x_data, y_data),
        method=method,
        iter_cb=lambda params, iter, resid, *args, **kwargs: optimization_callback(
            params, iter, resid, param_history, x_data, found_left, found_right, true_left, true_right, Pe),
        nan_policy='omit',
        # epsfcn = 1e-8
        # ftol = 1e-2
    )
    Pe_opt = reconstruct_Pe_list(result.params, Pe[0])

    df_history = process_results(param_history)
    return Pe_opt, df_history