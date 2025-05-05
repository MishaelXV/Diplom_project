import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from main_block.data import smooth_data, data_norm, generate_data, noize_data
from regression.global_models import model_ws, model_ms, predict_params
from regression.metrics import calculate_mae, calculate_mse, calculate_rmse, calculate_relative_mae

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


def find_growth_intervals(T_smooth, z_all, sigma, model_ws=None, model_ms=None, Pe0=None, A=None, N=None,
                          fixed_ws=None, fixed_ms=None):
    if fixed_ws is not None and fixed_ms is not None:
        window_size = int(round(fixed_ws))
        min_slope = fixed_ms
    else:
        window_size, min_slope = predict_params(Pe0, A, sigma, N, model_ws, model_ms)

    if window_size <= 1:
        window_size = 2
    if sigma == 0:
        return detect_growth_without_noise(T_smooth)
    else:
        return detect_growth_with_noise(T_smooth, z_all, window_size, min_slope)


def normalize_to_original_scale(boundaries, z_min, z_max):
    return [b * (z_max - z_min) + z_min for b in boundaries]


def get_interval_boundaries(filtered_intervals, z_all):
    start_indices = []
    end_indices = []
    left_boundaries = []
    right_boundaries = []
    in_interval = False

    for i in range(1, len(z_all)):
        if filtered_intervals[i] and not in_interval:
            start_indices.append(i - 1)
            left_boundaries.append(z_all[i - 1])
            in_interval = True
        elif not filtered_intervals[i] and in_interval:
            end_indices.append(i - 1)
            right_boundaries.append(z_all[i - 1])
            in_interval = False

    if end_indices and end_indices[-1] < len(z_all) - 1:
        end_indices[-1] = len(z_all) - 1
        right_boundaries[-1] = z_all[-1]

    return left_boundaries, right_boundaries, start_indices, end_indices


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


def merge_growth_intervals_by_gap(start_indices, end_indices, z_norm, max_gap=0.04):
    if len(start_indices) == 0:
        return [], []

    merged_starts = [start_indices[0]]
    merged_ends = [end_indices[0]]

    for i in range(1, len(start_indices)):
        gap = z_norm[start_indices[i]] - z_norm[merged_ends[-1]]
        if gap <= max_gap:
            merged_ends[-1] = end_indices[i]
        else:
            merged_starts.append(start_indices[i])
            merged_ends.append(end_indices[i])

    return merged_starts, merged_ends


def remove_short_intervals(start_indices, end_indices, z_norm, min_length=0.01):
    filtered_starts = []
    filtered_ends = []

    for start, end in zip(start_indices, end_indices):
        length = z_norm[end] - z_norm[start]
        if length >= min_length:
            filtered_starts.append(start)
            filtered_ends.append(end)

    return filtered_starts, filtered_ends


def get_boundaries(x_data, y_data_noize, Pe, N, sigma, A, model_ws=None, model_ms=None,
                   fixed_ws=None, fixed_ms=None):
    z_norm, T_noisy_norm = data_norm(x_data, y_data_noize)
    T_smooth = smooth_data(T_noisy_norm)
    filtered_intervals = find_growth_intervals(
        T_smooth, z_norm, sigma,
        model_ws=model_ws, model_ms=model_ms, Pe0=Pe[0], A=A, N=N,
        fixed_ws=fixed_ws, fixed_ms=fixed_ms
    )

    left, right, starts, ends = get_interval_boundaries(filtered_intervals, z_norm)
    starts, ends = merge_growth_intervals_by_gap(starts, ends, z_norm, max_gap=0.01)
    starts, ends = remove_short_intervals(starts, ends, z_norm, min_length=0.01)
    left_boundaries, right_boundaries = process_detected_boundaries(starts, ends, z_norm, x_data)
    return left_boundaries, right_boundaries


def plot_data(z_norm, T_smooth, start_indices, end_indices):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12})

    plt.figure(figsize=(10, 5))
    plt.plot(z_norm, T_smooth, label="Сглаженная кривая", linewidth=2, color='red')

    for start, end in zip(start_indices, end_indices):
        plt.axvspan(z_norm[start], z_norm[end], color='green', alpha=0.15)

    plt.xlabel("z/rw")
    plt.ylabel("θ")
    plt.title("Интервалы роста")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()


# тестовый пример
def main():
    boundary_dict = {'left': [0, 150, 300, 450],
                     'right': [100, 250, 350, 520]}
    Pe = [20000, 10000, 5000, 0]
    N = 300
    sigma = 0.001
    TG0 = 1
    atg = 0.0001
    A = 2

    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, N, sigma, A, model_ws, model_ms)
    print(found_left, found_right)

    z_norm, T_noisy_norm = data_norm(x_data, y_data_noize)
    T_smooth = smooth_data(T_noisy_norm)

    filtered_intervals = find_growth_intervals(
        T_smooth, z_norm, sigma,
        model_ws=model_ws, model_ms=model_ms, Pe0=Pe[0], A=A, N=N
    )

    left, right, starts, ends = get_interval_boundaries(filtered_intervals, z_norm)
    starts, ends = merge_growth_intervals_by_gap(starts, ends, z_norm, max_gap=0.01)
    starts, ends = remove_short_intervals(starts, ends, z_norm, min_length=0.01)

    plot_data(z_norm, T_smooth, starts, ends)

if __name__ == "__main__":
    main()