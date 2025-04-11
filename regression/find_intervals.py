import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from calculates_block.calculates import get_interval_boundaries
from calculates_block.data import smooth_data, generate_data
from regression.metrics import calculate_error_percentage, calculate_boundary_errors
from regression.optuna_search import predict_params, train_models

def calculate_window_slope(z_window, T_window):
    model = LinearRegression().fit(z_window, T_window)
    return model.coef_[0]


def detect_growth_without_noise(T_smooth):
    dTdz = np.gradient(T_smooth)
    return dTdz > 1e-6


def detect_growth_with_noise(T_smooth, z_all, window_size, min_slope):
    filtered_intervals = np.zeros_like(T_smooth, dtype=bool)
    for i in range(len(z_all) - window_size):
        z_window = np.array(z_all[i:i + window_size]).reshape(-1, 1)
        T_window = np.array(T_smooth[i:i + window_size])
        slope = calculate_window_slope(z_window, T_window)
        if slope > min_slope:
            filtered_intervals[i:i + window_size] = True
    return filtered_intervals


def find_growth_intervals(T_smooth, z_all, sigma, model_ws, model_ms, Pe0, A, N):
    window_size, min_slope = predict_params(Pe0, A, sigma, N, model_ws, model_ms)

    if sigma == 0:
        return detect_growth_without_noise(T_smooth)
    else:
        return detect_growth_with_noise(T_smooth, z_all, window_size, min_slope)


def filter_small_intervals(z_starts, z_ends, min_length):
    filtered_starts, filtered_ends = [], []
    for s, e in zip(z_starts, z_ends):
        if e - s >= min_length:
            filtered_starts.append(s)
            filtered_ends.append(e)
    return filtered_starts, filtered_ends


def merge_close_intervals(filtered_starts, filtered_ends, max_gap):
    if not filtered_starts:
        return np.array([]), np.array([])

    merged_starts = [filtered_starts[0]]
    merged_ends = [filtered_ends[0]]

    for i in range(1, len(filtered_starts)):
        if filtered_starts[i] - merged_ends[-1] <= max_gap:
            merged_ends[-1] = filtered_ends[i]
        else:
            merged_starts.append(filtered_starts[i])
            merged_ends.append(filtered_ends[i])

    return merged_starts, merged_ends


def convert_to_original_indices(merged_starts, merged_ends, z_norm):
    new_starts = [np.argmin(np.abs(z_norm - s)) for s in merged_starts]
    new_ends = [np.argmin(np.abs(z_norm - e)) for e in merged_ends]
    return np.array(new_starts), np.array(new_ends)


def postprocess_intervals(start_indices, end_indices, z_norm, min_interval_length=0.05, max_gap_length=0.03):
    if len(start_indices) == 0:
        return start_indices, end_indices

    z_starts = z_norm[start_indices]
    z_ends = z_norm[end_indices]

    filtered_starts, filtered_ends = filter_small_intervals(z_starts, z_ends, min_interval_length)
    if not filtered_starts:
        return np.array([]), np.array([])

    merged_starts, merged_ends = merge_close_intervals(filtered_starts, filtered_ends, max_gap_length)
    return convert_to_original_indices(merged_starts, merged_ends, z_norm)


def normalize_to_original_scale(boundaries, z_min, z_max):
    return [b * (z_max - z_min) + z_min for b in boundaries]


def process_detected_boundaries(start_indices, end_indices, z_norm, z_all):
    if len(start_indices) > 0:
        left = z_norm[start_indices]
        right = z_norm[end_indices]
    else:
        left, right = [], []

    z_min, z_max = z_all.min(), z_all.max()
    left_original = normalize_to_original_scale(left, z_min, z_max)
    right_original = normalize_to_original_scale(right, z_min, z_max)

    return left_original, right_original


def get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms):
    if len(boundary_dict['left']) != len(boundary_dict['right']):
        raise ValueError("Количество левых и правых границ должно совпадать")

    z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy = generate_data(
        boundary_dict['left'], boundary_dict['right'], Pe, 100000, TG0, atg, A, sigma, N)

    T_smooth = smooth_data(T_noisy_norm)
    filtered_intervals = find_growth_intervals(T_smooth, z_norm, sigma, model_ws, model_ms, Pe[0], A, N)
    left, right, starts, ends = get_interval_boundaries(filtered_intervals, z_norm)

    starts, ends = postprocess_intervals(starts, ends, z_norm)
    left_original, right_original = process_detected_boundaries(starts, ends, z_norm, z_all)

    return left_original, right_original, z_norm, T_true_norm, T_noisy_norm, T_smooth, starts, ends


def plot_data(z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices):
    plt.figure(figsize=(10, 5))
    plt.plot(z_norm, T_true_norm, label="Нормированная исходная кривая", marker='o', markersize=3, linestyle='-',
             color='blue')
    plt.plot(z_norm, T_noisy_norm, label="Нормированная шумная кривая", alpha=0.7, color='gray')
    plt.plot(z_norm, T_smooth, label="Нормированная сглаженная кривая", linewidth=2, color='red')

    for start, end in zip(start_indices, end_indices):
        plt.axvspan(z_norm[start], z_norm[end], color='green', alpha=0.2)

    plt.xlabel("z (нормированная)")
    plt.ylabel("T (нормированная)")
    plt.title("Нормированные интервалы роста")
    plt.legend()
    plt.show()


def load_training_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RELATIVE_PATH = os.path.join('regression', 'training_data.txt')
    FULL_PATH = os.path.join(BASE_DIR, RELATIVE_PATH)
    return pd.read_csv(FULL_PATH, sep='\t')


def print_metrics(true_left, true_right, found_left, found_right):
    error_percentage = calculate_error_percentage(true_left, true_right, found_left, found_right)
    total_error_mae, individual_errors = calculate_boundary_errors(true_left, true_right, found_left, found_right)
    left_errors = np.array(individual_errors['left_errors'])
    right_errors = np.array(individual_errors['right_errors'])
    total_error_mse = np.sum(left_errors ** 2) + np.sum(right_errors ** 2)

    print("\nРезультаты оценки точности:")
    print("=" * 40)
    print(f"Найденные левые границы: {found_left}")
    print(f"Найденные правые границы: {found_right}")
    print(f"Средняя относительная погрешность: {error_percentage:.2f}%")
    print(f"Суммарная абсолютная ошибка (MAE): {total_error_mae:.4f}")
    print(f"Суммарная квадратичная ошибка (MSE): {total_error_mse:.4f}")
    print("\nОшибки по границам:")
    print(f"Левые границы: {individual_errors['left_errors']}")
    print(f"Правые границы: {individual_errors['right_errors']}")
    print("=" * 40)

# тестовый пример
def main():
    boundary_dict = {'left': [0, 150, 300], 'right': [100, 250, 400]}
    Pe = [5000, 2000, 0]
    N = 60
    sigma = 0.001
    TG0 = 1
    atg = 0.0001
    A = 5

    df = load_training_data()
    model_ws, model_ms = train_models(df)

    results = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms)
    found_left, found_right, z_norm, T_true_norm, T_noisy_norm, T_smooth, starts, ends = results

    plot_data(z_norm, T_true_norm, T_noisy_norm, T_smooth, starts, ends)
    print_metrics(boundary_dict['left'], boundary_dict['right'], found_left, found_right)


if __name__ == "__main__":
    main()