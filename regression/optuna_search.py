import itertools
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from calculates_block.data import smooth_data, generate_data
from regression.find_intervals import get_interval_boundaries
from regression.metrics import calculate_boundary_errors
from joblib import Parallel, delayed

def evaluate_pipeline(regression_window_size, min_slope, Pe, A, N, boundary_dict, TG0=1, atg=0.0001):
    try:
        z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy = generate_data(
            boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)

        T_smooth = smooth_data(T_noisy_norm)
        filtered = np.zeros_like(T_smooth, dtype=bool)

        for i in range(len(z_norm) - regression_window_size):
            z_window = np.array(z_norm[i:i + regression_window_size]).reshape(-1, 1)
            T_window = T_smooth[i:i + regression_window_size]
            model = LinearRegression().fit(z_window, T_window)
            slope = model.coef_[0]
            if slope > min_slope:
                filtered[i:i + regression_window_size] = True

        found_left, found_right, *_ = get_interval_boundaries(filtered, z_norm)

        z_min, z_max = z_all.min(), z_all.max()
        found_left = [z_min + b * (z_max - z_min) for b in found_left]
        found_right = [z_min + b * (z_max - z_min) for b in found_right]

        if len(found_left) != len(boundary_dict['left']):
            return None

        mae, _ = calculate_boundary_errors(boundary_dict['left'], boundary_dict['right'], found_left, found_right)
        return mae
    except Exception as e:
        print(f"Ошибка в evaluate_pipeline: {e}")
        return None


def objective(trial, Pe, A, sigma, N, boundary_dict, n_repeats=5):
    window_size = trial.suggest_int("window_size", 1, 21)
    min_slope = trial.suggest_float("min_slope", 0.01, 1.2)

    errors = []
    for _ in range(n_repeats):
        error = evaluate_pipeline(window_size, min_slope, Pe, A, sigma, N, boundary_dict)
        if error is not None:
            errors.append(error)

    if not errors:
        return float("inf")

    return np.mean(errors)


def optimize_params(Pe, A, sigma, N, boundary_dict, n_repeats=5):
    study = optuna.create_study(direction="minimize")

    study.optimize(lambda trial: objective(trial, Pe, A, sigma, N, boundary_dict, n_repeats), n_trials=50,
                   show_progress_bar=False)

    return study.best_params


def collect_training_data():
    training_data = []

    param_grid = list(itertools.product(
        np.linspace(2000, 20000, 18),  # Pe[0]
        np.linspace(1, 11, 10),      # A
        np.linspace(0.0001, 0.01, 20),     # sigma
        [50, 100, 200, 300, 400, 500],            # N
    ))

    boundary_dict = {'left': [0, 150, 300], 'right': [100, 250, 400]}


    def process_params(Pe0, A, sigma, N):
        Pe = [Pe0, 1000, 0]
        best_params = optimize_params(Pe, A, sigma, N, boundary_dict)

        if best_params:
            row = {
                "Pe0": Pe0,
                "A": A,
                "sigma": sigma,
                "N": N,
                "window_size": best_params["window_size"],
                "min_slope": best_params["min_slope"]
            }
            return row
        return None

    results = Parallel(n_jobs=-1)(delayed(process_params)(Pe0, A, sigma, N) for Pe0, A, sigma, N in tqdm(param_grid))

    training_data = [res for res in results if res is not None]
    return pd.DataFrame(training_data)