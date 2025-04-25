import numpy as np
from calculates_block.main_functions import main_func
from scipy.signal import medfilt
from pathlib import Path

def save_temperature_values(T_all, filename):
    project_root = Path(__file__).parent.parent.parent

    target_dir = project_root / "stability_tests" / "data" / "values"

    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / filename

    with open(file_path, "w") as file:
        for value in T_all:
            file.write(f"{value}\n")


def data_norm(x_data, y_data, y_data_noize):
    z_min, z_max = x_data.min(), x_data.max()
    z_norm = (x_data - z_min) / (z_max - z_min)

    T_min, T_max = y_data.min(), y_data.max()
    T_true_norm = (y_data - T_min) / (T_max - T_min)
    T_noisy_norm = (y_data_noize - T_min) / (T_max - T_min)

    return z_norm, T_true_norm, T_noisy_norm


def generate_data(left_boundaries, right_boundaries, Pe, TG0, atg, A, N):
    x_data = np.linspace(min(left_boundaries), max(right_boundaries), N)
    y_data = main_func(x_data, TG0, atg, A, Pe, left_boundaries, right_boundaries)

    return x_data, y_data


def noize_data(y_data, sigma):
    y_data_noize = y_data + np.random.normal(0, sigma, y_data.size)
    return y_data_noize


def smooth_data(T_all):
    window_size = 71
    smoothed = medfilt(T_all, kernel_size = window_size)
    return smoothed