import optuna
import numpy as np
from lmfit import Parameters
from main_block.main_functions import reconstruct_Pe_list, main_func
from optimizator.optimizer import calculate_deviation_metric
from optimizator.process import process_results

def calculate_deltas_from_weights(weights, known_pe1):
    total = sum(weights)
    return [w / total * known_pe1 for w in weights]


def create_params_from_deltas(deltas):
    params = Parameters()
    for i, delta in enumerate(deltas):
        params.add(f"delta_{i}", value=delta)
    return params


def objective_function(trial, x_data, y_data_noize, known_pe1,
                       found_left, found_right, true_left, true_right, Pe_true,
                       TG0, atg, A, param_history):

    n_layers = len(found_left) - 1
    weights = [trial.suggest_float(f"w_{i}", 0.01, 1.0) for i in range(n_layers)]
    deltas = calculate_deltas_from_weights(weights, known_pe1)
    params = create_params_from_deltas(deltas)

    Pe_opt = reconstruct_Pe_list(params, known_pe1)
    y_pred = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)
    residuals = y_pred - y_data_noize
    loss = float(np.sum(residuals ** 2))

    deviation_metric = calculate_deviation_metric(x_data, found_left, found_right, true_left, true_right, Pe_true, Pe_opt)

    param_values = {f"w_{i}": w for i, w in enumerate(weights)}
    param_values["Pe"] = Pe_opt
    param_values["residuals"] = loss

    param_history.append((param_values, deviation_metric, trial.number))

    return loss


def run_bayes_optimization(x_data, y_data_noize, found_left, found_right,
                           true_left, true_right, Pe, TG0, atg, A, n_trials=200):
    known_pe1 = Pe[0]
    param_history = []

    def wrapped_objective(trial):
        return objective_function(trial, x_data, y_data_noize, known_pe1,
                                  found_left, found_right, true_left, true_right, Pe,
                                  TG0, atg, A, param_history)

    study = optuna.create_study(direction='minimize')
    study.optimize(wrapped_objective, n_trials=n_trials)

    n_layers = len(found_left) - 1
    best_weights = [study.best_params[f"w_{i}"] for i in range(n_layers)]
    deltas = calculate_deltas_from_weights(best_weights, known_pe1)
    best_params = create_params_from_deltas(deltas)

    Pe_opt = reconstruct_Pe_list(best_params, known_pe1)
    df_history = process_results(param_history)
    return Pe_opt, df_history