import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed
from tqdm import tqdm
from calculates_block.data import generate_data, noize_data
from regression.find_intervals import get_boundaries
from regression.metrics import calculate_mae, calculate_relative_mae, calculate_rmse, calculate_mse
from regression.optuna_search import load_training_data

def train_models(df, model_type='GradientBoosting'):
    X = df[["Pe0", "A", "sigma", "N"]]
    y_ws = df["window_size"]
    y_ms = df["min_slope"]

    if model_type == 'GradientBoosting':
        model_ws = GradientBoostingRegressor(random_state=42)
        model_ms = GradientBoostingRegressor(random_state=42)
    elif model_type == 'RandomForest':
        model_ws = RandomForestRegressor(random_state=42)
        model_ms = RandomForestRegressor(random_state=42)
    elif model_type == 'SVR':
        model_ws = make_pipeline(StandardScaler(), SVR())
        model_ms = make_pipeline(StandardScaler(), SVR())
    elif model_type == 'MLP':
        model_ws = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000, random_state=42))
        model_ms = make_pipeline(StandardScaler(), MLPRegressor(max_iter=1000, random_state=42))
    elif model_type == 'LinearRegression':
        model_ws = make_pipeline(StandardScaler(), LinearRegression())
        model_ms = make_pipeline(StandardScaler(), LinearRegression())

    model_ws.fit(X, y_ws)
    model_ms.fit(X, y_ms)
    return model_ws, model_ms


def _single_evaluation(predict_ws, predict_ms, boundary_dict, Pe, N, sigma, TG0, atg, A):
    start_time = time.time()
    use_fixed = not hasattr(predict_ws, "predict")

    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    if use_fixed:
        left, right = get_boundaries( x_data, y_data, y_data_noize, Pe, N, sigma, A,
            fixed_ws=predict_ws, fixed_ms=predict_ms
        )
    else:
        left, right = get_boundaries( x_data, y_data, y_data_noize, Pe, N, sigma, A,
            model_ws=predict_ws, model_ms=predict_ms
        )

    return {
        'total_mae': calculate_mae(boundary_dict['left'], boundary_dict['right'], left, right),
        'relative_mae': calculate_relative_mae(boundary_dict['left'], boundary_dict['right'], left, right),
        'total_mse': calculate_mse(boundary_dict['left'], boundary_dict['right'], left, right),
        'total_rmse': calculate_rmse(boundary_dict['left'], boundary_dict['right'], left, right),
        'time': time.time() - start_time
    }


def evaluate_boundaries(predict_ws, predict_ms, boundary_dict, Pe, N, sigma, TG0, atg, A, n_runs):
    results = Parallel(n_jobs=6)(
        delayed(_single_evaluation)(
            predict_ws, predict_ms, boundary_dict, Pe, N, sigma, TG0, atg, A
        ) for _ in tqdm(range(n_runs), desc=f"Оценка ({getattr(predict_ws, '__class__', type(predict_ws)).__name__})", leave=False)
    )
    aggregated = {key: [r[key] for r in results] for key in results[0]}
    return {f'mean_{k}': np.mean(v) for k, v in aggregated.items()}


def compare_models(df, boundary_dict, Pe, N, sigma, TG0, atg, A, n_runs):
    models = ['MeanBaseline', 'LinearRegression', 'RandomForest', 'GradientBoosting', 'SVR', 'MLP']
    results = []

    for model_type in tqdm(models, desc="Сравнение моделей"):
        print(f"\nОценка модели {model_type}...")

        if model_type == 'MeanBaseline':
            window_size = df["window_size"].mean()
            min_slope = df["min_slope"].mean()
            predict_ws = window_size
            predict_ms = min_slope
            train_time = 0.0
        else:
            train_start = time.time()
            model_ws, model_ms = train_models(df, model_type)
            predict_ws = model_ws
            predict_ms = model_ms
            train_time = time.time() - train_start

        metrics = evaluate_boundaries(
            predict_ws, predict_ms, boundary_dict, Pe, N, sigma, TG0, atg, A, n_runs)

        results.append({
            'model': model_type,
            'train_time': train_time,
            **metrics
        })

    return pd.DataFrame(results)


def plot_results(results_df):
    sns.set(style="whitegrid")
    palette_mae = sns.color_palette("crest", len(results_df))
    palette_time = sns.color_palette("flare", len(results_df))

    results_mae_sorted = results_df.sort_values('mean_relative_mae')
    results_time_sorted = results_df.sort_values('mean_time')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    bars1 = ax1.bar(results_mae_sorted['model'], results_mae_sorted['mean_relative_mae'], color=palette_mae)
    ax1.set_title('Относительная ошибка MAE (%)', fontsize=14)
    ax1.set_ylabel('MAE (%)')
    ax1.grid(False)
    ax1.tick_params(axis='x', rotation=30)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{bar.get_height():.2f}%", ha='center', fontsize=9)

    ax2 = axes[1]
    bars2 = ax2.bar(results_time_sorted['model'], results_time_sorted['mean_time'], color=palette_time)
    ax2.set_title('Среднее время на реализацию', fontsize=14)
    ax2.set_ylabel('Время (сек)')
    ax2.grid(False)
    ax2.tick_params(axis='x', rotation=30)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{bar.get_height():.3f}", ha='center', fontsize=9)

    plt.suptitle("Сравнение моделей по точности (%) и скорости", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Models_test.png", format="png", dpi=300)
    plt.show()


def main():
    boundary_dict = {'left': [0, 150, 300], 'right': [100, 250, 400]}
    Pe = [2000, 1000, 0]
    N = 300
    sigma = 0.001
    TG0 = 1
    atg = 0.0001
    A = 10
    n_runs = 1

    df = load_training_data()
    results = compare_models(df, boundary_dict, Pe, N, sigma, TG0, atg, A, n_runs)

    results_sorted = results.sort_values('mean_relative_mae')
    print("\nРезультаты сравнения моделей (относительная MAE):")
    print(results_sorted[['model', 'mean_relative_mae', 'mean_time']].to_string(index=False))

    plot_results(results)

if __name__ == "__main__":
    main()