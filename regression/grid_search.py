import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from main_block.data import smooth_data, generate_data
from regression.find_intervals import get_interval_boundaries
from regression.metrics import calculate_boundary_errors

def evaluate_pipeline(regression_window_size, min_slope, Pe, A, sigma, N, boundary_dict, TG0=1, atg=0.0001, zInf=100000):
    try:
        z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy = generate_data(
            boundary_dict['left'], boundary_dict['right'], Pe, zInf, TG0, atg, A, sigma, N)

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


def optimize_params_grid(Pe, A, sigma, N, boundary_dict):
    best_error = float("inf")
    best_params = None

    for window_size in range(3, 22):
        for min_slope in np.linspace(0.01, 0.5, 20):
            error = evaluate_pipeline(window_size, min_slope, Pe, A, sigma, N, boundary_dict)
            if error is not None and error < best_error:
                best_error = error
                best_params = {"window_size": window_size, "min_slope": min_slope}

    return best_params


def collect_training_data():
    training_data = []

    param_grid = list(itertools.product(
        np.linspace(500, 5000, 5),
        np.linspace(1, 10, 5),
        np.linspace(0.0001, 0.01, 10),
        [50, 100, 200],
    ))

    boundary_dict = {'left': [0, 150, 300], 'right': [100, 250, 400]}

    for Pe0, A, sigma, N in tqdm(param_grid):
        Pe = [Pe0, 1000, 0]
        best_params = optimize_params_grid(Pe, A, sigma, N, boundary_dict)

        if best_params:
            row = {
                "Pe0": Pe0,
                "A": A,
                "sigma": sigma,
                "N": N,
                "window_size": best_params["window_size"],
                "min_slope": best_params["min_slope"]
            }
            training_data.append(row)

    return pd.DataFrame(training_data)


def train_models(df):
    X = df[["Pe0", "A", "sigma", "N"]]
    y_ws = df["window_size"]
    y_ms = df["min_slope"]

    X_train, X_test, y_ws_train, y_ws_test = train_test_split(X, y_ws, test_size=0.2, random_state=42)
    _, _, y_ms_train, y_ms_test = train_test_split(X, y_ms, test_size=0.2, random_state=42)

    model_ws = GradientBoostingRegressor().fit(X_train, y_ws_train)
    model_ms = GradientBoostingRegressor().fit(X_train, y_ms_train)

    return model_ws, model_ms


def predict_params(Pe0, A, sigma, N, model_ws, model_ms):
    x_input = pd.DataFrame([[Pe0, A, sigma, N]], columns=["Pe0", "A", "sigma", "N"])
    ws = model_ws.predict(x_input)[0]
    ms = model_ms.predict(x_input)[0]
    return int(round(ws)), round(ms, 3)


if __name__ == "__main__":
    df = collect_training_data()
    df.to_csv("training_data_grid.txt", sep="\t", index=False)
    print(df)
