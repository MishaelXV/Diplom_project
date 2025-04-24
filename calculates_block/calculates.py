import numpy as np
from calculates_block.main_functions import TsGLin

def debit(Pe, lw=0.6, rw=0.1, cw=4200, row=1000):
    return 24 * 3600 * (Pe * lw * np.pi * rw) / (cw * row)


def calculate_temperatures(a, left_boundary, right_boundary, N, TG0, atg, A, b_values, TsGLin_array):
    z_all = []
    T_all = []

    for j in range(a):
        z = np.linspace(left_boundary[j], right_boundary[j], N)
        T_list = [TsGLin([z_val], 100000, TG0, atg, A, b_values[j], left_boundary[j], TsGLin_array[j]) for z_val in z]
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


def add_noise_to_temperature(T_all, sigma):
    noise = np.random.normal(loc=0, scale=sigma, size=len(T_all))
    return np.array(T_all) + noise


def prepare_dataframe(df_history):
    df_history['Невязка'] = df_history['Невязка'].apply(round_mantissa)
    df_history.reset_index(inplace=True)
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history.to_dict('records')


def prepare_dataframe_2(df_history):
    df_history = df_history.reset_index()
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history


def calculate_temperature(left_boundary, right_boundary, Pe, N, TsGLin_array, TG0, atg, A):
    z_all = []
    T_all = []

    for j in range(len(Pe)):
        z = np.linspace(left_boundary[j], right_boundary[j], N)
        T_list = [TsGLin([z_val], 100000, TG0, atg, A, Pe[j], left_boundary[j], TsGLin_array[j]) for z_val in z]
        T = [item[0] for item in T_list]

        z_all.extend(z)
        T_all.extend(T)

        if j < len(Pe) - 1:
            y_value = TsGLin_array[j + 1]
            start_point = right_boundary[j]
            end_point = left_boundary[j + 1]
            z_horizontal = np.linspace(start_point, end_point, N)
            T_all.extend([y_value] * len(z_horizontal))
            z_all.extend(z_horizontal)

    return z_all, T_all


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


