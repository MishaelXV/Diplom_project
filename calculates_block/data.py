import numpy as np
from calculates_block.main_functions import main_func
from scipy.signal import medfilt

def generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    z_all = np.linspace(b[0], c[-1], N)
    T_true = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, z_all, zInf, TG0, atg, A, Pe, b, c)

    # Накладываем шум до нормировки
    T_noisy = T_true + np.random.normal(0, sigma, z_all.size)

    # Нормировка данных
    z_min, z_max = z_all.min(), z_all.max()
    z_norm = (z_all - z_min) / (z_max - z_min)

    T_min, T_max = T_noisy.min(), T_noisy.max()
    T_true_norm = (T_true - T_min) / (T_max - T_min)
    T_noisy_norm = (T_noisy - T_min) / (T_max - T_min)

    return z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy


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