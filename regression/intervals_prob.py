import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from calculates_block.calculates import get_interval_boundaries
from calculates_block.data import smooth_data, generate_data
from regression.metrics import calculate_error_percentage, calculate_boundary_errors
from regression.trainer import predict_params, train_models

def find_growth_intervals(T_smooth, z_all, sigma, model_ws, model_ms, Pe0, A, N):
    filtered_intervals = np.zeros_like(T_smooth, dtype=bool)

    window_size, min_slope = predict_params(Pe0, A, sigma, N, model_ws, model_ms)

    if sigma == 0:
        dTdz = np.gradient(T_smooth, z_all)
        threshold = np.max(np.abs(dTdz)) * 1e-6
        filtered_intervals = dTdz > threshold
    else:
        for i in range(len(z_all) - window_size):
            z_window = np.array(z_all[i:i + window_size]).reshape(-1, 1)
            T_window = np.array(T_smooth[i:i + window_size])

            model = LinearRegression().fit(z_window, T_window)
            slope = model.coef_[0]

            if slope > min_slope:
                filtered_intervals[i:i + window_size] = True

    return filtered_intervals

def get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms):
    left_boundary = boundary_dict['left']
    right_boundary = boundary_dict['right']

    if len(left_boundary) != len(right_boundary):
        raise ValueError("Количество левых и правых границ должно совпадать")

    z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy = generate_data(
        left_boundary, right_boundary, Pe, 100000, TG0, atg, A, sigma, N)

    T_smooth = smooth_data(T_noisy_norm)

    filtered_intervals = find_growth_intervals(T_smooth, z_norm, sigma, model_ws, model_ms, Pe[0], A, N)

    left_boundaries, right_boundaries, start_indices, end_indices = get_interval_boundaries(filtered_intervals, z_norm)

    z_min, z_max = z_all.min(), z_all.max()
    left_boundaries_original = [b * (z_max - z_min) + z_min for b in left_boundaries]
    right_boundaries_original = [b * (z_max - z_min) + z_min for b in right_boundaries]

    return left_boundaries_original, right_boundaries_original, z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices

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

if __name__ == "__main__":
    boundary_dict = {'left': [0, 150, 300],
                     'right': [100, 250, 400]}
    Pe = [5000, 2000, 0]
    N = 150
    sigma = 0.001
    TG0 = 1
    atg = 0.0001
    A = 10

    df = pd.read_csv("/Users/macbookmike_1/PycharmProjects/PythonProject/regression/training_data.txt", sep="\t")
    model_ws, model_ms = train_models(df)

    found_left, found_right, z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices = get_boundaries(
        boundary_dict, Pe, N, sigma, TG0, atg, A, model_ws, model_ms)

    plot_data(z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices)

    error_percentage = calculate_error_percentage(boundary_dict['left'], boundary_dict['right'], found_left, found_right)

    total_error_mae, individual_errors = calculate_boundary_errors(
        boundary_dict['left'], boundary_dict['right'], found_left, found_right
    )

    left_errors = np.array(individual_errors['left_errors'])
    right_errors = np.array(individual_errors['right_errors'])
    total_error_mse = np.sum(left_errors ** 2) + np.sum(right_errors ** 2)

    # Вывод результатов в терминал
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