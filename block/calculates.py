import numpy as np
from block.block import TsGLin
from optimizator.optimizer import run_optimization

def calculate_temperatures(a, left_boundary, right_boundary, N, TG0, atg, A, b_values, TsGLin_array):
    z_all = []
    T_all = []

    for j in range(a):
        z = np.linspace(left_boundary[j], right_boundary[j], N)
        T_list = [TsGLin([z_val], 100000, TG0, atg, A, b_values[j], left_boundary[j], TsGLin_array[j]) for z_val in
                  z]
        T = [item[0] for item in T_list]

        z_all.extend(z)
        T_all.extend(T)

        if j < a - 1:
            y_value = TsGLin_array[j + 1]
            start_point = right_boundary[j] + 1e-6
            end_point = left_boundary[j + 1]
            z_horizontal = np.linspace(start_point, end_point, N)
            T_all.extend([y_value] * len(z_horizontal))
            z_all.extend(z_horizontal)

    return z_all, T_all


def round_mantissa(value, decimals=5):
    formatted = f"{value:.{decimals}e}"
    return float(formatted)


def perform_optimization(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N):
    return run_optimization(left_boundary, right_boundary, b_values, 100000, TG0, atg, A, sigma, N)


def add_noise_to_temperature(T_all, sigma):
    noise = np.random.normal(loc=0, scale=sigma, size=len(T_all))
    return np.array(T_all) + noise