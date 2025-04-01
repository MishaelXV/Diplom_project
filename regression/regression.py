import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from calculates_block.calculates import TsGLin, calculate_TsGLin_array
from sklearn.linear_model import LinearRegression


def calculate_temperature(left_boundary, right_boundary, Pe, N, TsGLin_array, TG0, atg, A):
    z_all = []
    T_all = []

    # Находим общий диапазон z для нормировки
    z_min = min(left_boundary)
    z_max = max(right_boundary)
    z_span = z_max - z_min

    for j in range(len(Pe)):
        # Нормируем границы текущего интервала
        z_norm_left = (left_boundary[j] - z_min) / z_span
        z_norm_right = (right_boundary[j] - z_min) / z_span

        z = np.linspace(z_norm_left, z_norm_right, N)
        T_list = [TsGLin([z_val * z_span + z_min], 100000, TG0, atg, A, Pe[j],
                         left_boundary[j], TsGLin_array[j]) for z_val in z]
        T = [item[0] for item in T_list]

        z_all.extend(z)
        T_all.extend(T)

        if j < len(Pe) - 1:
            y_value = TsGLin_array[j + 1]
            start_point = z_norm_right + 1e-6 / z_span
            end_point = (left_boundary[j + 1] - z_min) / z_span
            z_horizontal = np.linspace(start_point, end_point, N)
            T_all.extend([y_value] * len(z_horizontal))
            z_all.extend(z_horizontal)

    return z_all, T_all


def smooth_data(T_all, sigma):
    # Накладываем шум ДО нормировки температуры
    noise = np.random.normal(0, sigma, size=len(T_all))
    T_noisy = np.array(T_all) + noise

    # Нормируем температуру к [0,1] после добавления шума
    T_min, T_max = np.min(T_noisy), np.max(T_noisy)
    T_span = T_max - T_min if T_max != T_min else 1
    T_noisy_normalized = (T_noisy - T_min) / T_span

    # Сглаживаем уже нормированные данные
    return medfilt(T_noisy_normalized, kernel_size=57)


def find_growth_intervals(T_smooth, z_all, sigma):
    filtered_intervals = np.zeros_like(T_smooth, dtype=bool)

    if sigma == 0:
        dTdz = np.gradient(T_smooth, z_all)
        filtered_intervals = dTdz > 1e-10
    else:
        window_size = 15
        min_slope = 0.7

        for i in range(len(z_all) - window_size):
            z_window = np.array(z_all[i:i + window_size]).reshape(-1, 1)
            T_window = np.array(T_smooth[i:i + window_size])

            model = LinearRegression().fit(z_window, T_window)
            slope = model.coef_[0]

            if slope > min_slope:
                filtered_intervals[i:i + window_size] = True

    return filtered_intervals


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


def plot_results(z_all, T_all, T_noisy, T_smooth, start_indices, end_indices, found_left, found_right):
    plt.figure(figsize=(14, 7))

    # Все z уже нормированы к [0,1], поэтому просто отображаем
    plt.plot(z_all, T_all, label="Исходная температура", color='blue', linewidth=1)
    plt.plot(z_all, T_noisy, label="Зашумленные данные", alpha=0.4, color='lightblue')
    plt.plot(z_all, T_smooth, label="Сглаженные данные", color='red', linewidth=1.5)

    # Остальной код визуализации без изменений
    for start, end in zip(start_indices, end_indices):
        plt.axvspan(z_all[start], z_all[end], color='green', alpha=0.2,
                    label="Интервал роста" if start == start_indices[0] else "")

    for x in found_left:
        plt.axvline(x=x, color='lime', linestyle='--', linewidth=1, alpha=0.7,
                    label="Левая граница" if x == found_left[0] else "")
    for x in found_right:
        plt.axvline(x=x, color='darkgreen', linestyle='--', linewidth=1, alpha=0.7,
                    label="Правая граница" if x == found_right[0] else "")

    plt.xlabel("Нормированная координата z ∈ [0,1]", fontsize=12)
    plt.ylabel("Нормированная температура T ∈ [0,1]", fontsize=12)
    plt.title("Анализ интервалов роста (нормированные данные)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()


def get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A):
    left_boundary = boundary_dict['left']
    right_boundary = boundary_dict['right']

    if len(left_boundary) != len(right_boundary):
        raise ValueError("Количество левых и правых границ должно совпадать")

    TsGLin_init = 0
    TsGLin_array = calculate_TsGLin_array(right_boundary, 100000, TG0, atg, A, Pe, left_boundary, TsGLin_init)
    TsGLin_array = [float(value) for value in TsGLin_array]

    z_all, T_all = calculate_temperature(left_boundary, right_boundary, Pe, N, TsGLin_array, TG0, atg, A)
    T_noisy = np.array(T_all) + np.random.normal(0, sigma, size=len(T_all))  # Сохраняем зашумленные данные
    T_smooth = medfilt(T_noisy, kernel_size=57)
    filtered_intervals = find_growth_intervals(T_smooth, z_all, sigma)
    left_boundaries, right_boundaries, start_indices, end_indices = get_interval_boundaries(filtered_intervals, z_all)

    # Строим график
    plot_results(z_all, T_all, T_noisy, T_smooth, start_indices, end_indices, left_boundaries, right_boundaries)

    return left_boundaries, right_boundaries


if __name__ == "__main__":
    # Параметры для демонстрации
    boundary_dict = {'left': [0, 250, 400, 550],
                     'right': [200, 350, 500, 600]}
    Pe = [3000, 1000, 500, 0]
    N = 100  # Увеличим количество точек для гладкости
    sigma = 0.005  # Уровень шума
    TG0 = 1
    atg = 0.0001
    A = 5

    # Запуск анализа
    found_left, found_right = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A)

    # Вывод результатов
    print("\nРезультаты анализа:")
    print(f"Найденные левые границы: {found_left}")
    print(f"Найденные правые границы: {found_right}")