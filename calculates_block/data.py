import numpy as np
from calculates_block.main_functions import main_func
from scipy.signal import medfilt
from pathlib import Path

def save_temperature_values(T_all, filename):
    project_root = Path(__file__).parent.parent.parent

    target_dir = project_root / "tests" / "data" / "values"

    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    with open(file_path, "w") as file:
        for value in T_all:
            file.write(f"{value}\n")


def generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    z_all = np.linspace(b[0], c[-1], N)
    T_true = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, z_all, zInf, TG0, atg, A, Pe, b, c)

    T_noisy = T_true + np.random.normal(0, sigma, z_all.size)

    z_min, z_max = z_all.min(), z_all.max()
    z_norm = (z_all - z_min) / (z_max - z_min)

    T_min, T_max = T_noisy.min(), T_noisy.max()
    T_true_norm = (T_true - T_min) / (T_max - T_min)
    T_noisy_norm = (T_noisy - T_min) / (T_max - T_min)

    return z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy


def generate_data_optim(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    x_data = np.linspace(b[0], c[-1], N)
    y_data = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, x_data, zInf, TG0, atg, A, Pe, b, c) + np.random.normal(0, sigma, x_data.size)
    return x_data, y_data


def smooth_data(T_all):
    n_points = len(T_all)

    if n_points < 50:
        window_size = 3
    elif n_points < 200:
        window_size = 21
    elif n_points < 1000:
        window_size = min(51, int(n_points * 0.1) // 2 * 2 + 1)
    else:
        window_size = 101

    window_size = min(window_size, n_points)
    window_size = window_size if window_size % 2 == 1 else window_size - 1

    return medfilt(T_all, kernel_size=window_size)