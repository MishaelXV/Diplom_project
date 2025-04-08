import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from calculates_block.calculates import TsGLin, calculate_TsGLin_array
from sklearn.linear_model import LinearRegression

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
            start_point = right_boundary[j] + 1e-6
            end_point = left_boundary[j + 1]
            z_horizontal = np.linspace(start_point, end_point, N)
            T_all.extend([y_value] * len(z_horizontal))
            z_all.extend(z_horizontal)

    return z_all, T_all


def smooth_data(T_all, sigma):
    noise = np.random.normal(0, sigma, size=len(T_all))
    T_noisy = np.array(T_all) + noise
    return medfilt(T_noisy, kernel_size=57)


def find_growth_intervals(T_smooth, z_all, sigma):
    filtered_intervals = np.zeros_like(T_smooth, dtype=bool)

    if sigma == 0:
        dTdz = np.gradient(T_smooth, z_all)
        filtered_intervals = dTdz > 1e-10
    else:
        window_size = 5
        min_slope = 0.0009

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


def plot_data(z_all, T_all, T_noisy, T_smooth, start_indices, end_indices):
    plt.figure(figsize=(10, 5))
    plt.plot(z_all, T_all, label="Исходная кривая", marker='o', markersize=3, linestyle='-', color='blue')
    plt.plot(z_all, T_noisy, label="Шумная кривая", alpha=0.7, color='gray')
    plt.plot(z_all, T_smooth, label="Сглаженная кривая", linewidth=2,color='red')

    for start, end in zip(start_indices, end_indices):
        plt.axvspan(z_all[start], z_all[end], color='green', alpha=0.2)

    plt.xlabel("z")
    plt.ylabel("T")
    plt.title("Интервалы роста")
    plt.legend()
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
    T_smooth = smooth_data(T_all, sigma)
    filtered_intervals = find_growth_intervals(T_smooth, z_all, sigma)
    left_boundaries, right_boundaries, start_indices, end_indices = get_interval_boundaries(filtered_intervals, z_all)
    return left_boundaries, right_boundaries


def calculate_error_percentage(true_left, true_right, found_left, found_right):
    if not all(isinstance(lst, (list, tuple)) for lst in [true_left, true_right, found_left, found_right]):
        print("Ошибка: все входные данные должны быть списками или кортежами")
        return None

    if len(true_left) != len(found_left) or len(true_right) != len(found_right):
        print("Предупреждение: количество найденных границ не совпадает с истинными")
        return None

    if not true_left or not true_right:
        print("Ошибка: пустые списки границ")
        return None

    def calculate_errors(true_vals, found_vals):
        errors = []
        for t, f in zip(true_vals, found_vals):
            try:
                if t == 0:
                    errors.append(abs(f) * 100)
                else:
                    errors.append((abs(t - f) / t) * 100)
            except TypeError:
                print(f"Ошибка: нечисловые значения в данных ({t}, {f})")
                return None
        return errors

    left_errors = calculate_errors(true_left, found_left)
    right_errors = calculate_errors(true_right, found_right)

    if left_errors is None or right_errors is None:
        return None

    total_errors = left_errors + right_errors
    avg_error = sum(total_errors) / len(total_errors)

    return avg_error


if __name__ == "__main__":
    boundary_dict = {'left': [100, 250, 400], 'right': [200, 350, 500]}
    Pe = [3000, 1000, 0]
    N = 50
    sigma = 0.001
    TG0 = 1
    atg = 0.0001
    A = 5

    # Получаем границы и данные для визуализации
    found_left, found_right = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A)

    # Для plot_data нам нужно получить промежуточные данные:
    TsGLin_init = 0
    TsGLin_array = calculate_TsGLin_array(boundary_dict['right'], 100000, TG0, atg, A, Pe, boundary_dict['left'],
                                          TsGLin_init)
    TsGLin_array = [float(value) for value in TsGLin_array]

    # Вычисляем температурные профили
    z_all, T_all = calculate_temperature(boundary_dict['left'], boundary_dict['right'], Pe, N, TsGLin_array, TG0, atg,
                                         A)

    # Добавляем шум и сглаживаем
    T_noisy = np.array(T_all) + np.random.normal(0, sigma, size=len(T_all))
    T_smooth = medfilt(T_noisy, kernel_size=57)

    # Находим интервалы роста
    filtered_intervals = find_growth_intervals(T_smooth, z_all, sigma)
    left_bounds, right_bounds, start_indices, end_indices = get_interval_boundaries(filtered_intervals, z_all)

    # Визуализация
    plot_data(z_all, T_all, T_noisy, T_smooth, start_indices, end_indices)

    # Вычисляем процентную погрешность
    error_percentage = calculate_error_percentage(boundary_dict['left'], boundary_dict['right'], found_left,
                                                  found_right)

    print("Найденные левые границы:", found_left)
    print("Найденные правые границы:", found_right)
    print(f"Средняя относительная погрешность: {error_percentage:.2f}%")

