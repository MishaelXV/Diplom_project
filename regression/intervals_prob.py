import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from calculates_block.calculates import main_func
from sklearn.linear_model import LinearRegression


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


def find_growth_intervals(T_smooth, z_all, sigma):
    filtered_intervals = np.zeros_like(T_smooth, dtype=bool)

    if sigma == 0:
        dTdz = np.gradient(T_smooth, z_all)
        threshold = np.max(np.abs(dTdz)) * 1e-6
        filtered_intervals = dTdz > threshold
    else:
        window_size = 5
        min_slope = 0.4

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


def generate_data_2(b, c, Pe, zInf, TG0, atg, A, sigma, N):
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


def get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A):
    left_boundary = boundary_dict['left']
    right_boundary = boundary_dict['right']

    if len(left_boundary) != len(right_boundary):
        raise ValueError("Количество левых и правых границ должно совпадать")

    z_norm, T_true_norm, T_noisy_norm, z_all, T_true, T_noisy = generate_data_2(
        left_boundary, right_boundary, Pe, 100000, TG0, atg, A, sigma, N)

    T_smooth = smooth_data(T_noisy_norm)
    filtered_intervals = find_growth_intervals(T_smooth, z_norm, sigma)
    left_boundaries, right_boundaries, start_indices, end_indices = get_interval_boundaries(filtered_intervals, z_norm)

    # Преобразуем границы обратно в исходные координаты
    z_min, z_max = z_all.min(), z_all.max()
    left_boundaries_original = [b * (z_max - z_min) + z_min for b in left_boundaries]
    right_boundaries_original = [b * (z_max - z_min) + z_min for b in right_boundaries]

    return left_boundaries_original, right_boundaries_original, z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices


def validate_inputs(true_left, true_right, found_left, found_right):
    if not all(isinstance(lst, (list, tuple)) for lst in [true_left, true_right, found_left, found_right]):
        print("Ошибка: все входные данные должны быть списками или кортежами")
        return False

    if len(true_left) != len(found_left) or len(true_right) != len(found_right):
        print("Предупреждение: количество найденных границ не совпадает с истинными")
        return False

    if not true_left or not true_right:
        print("Ошибка: пустые списки границ")
        return False

    return True


def calculate_single_boundary_errors(true_vals, found_vals):
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


def calculate_average_error(left_errors, right_errors):
    if left_errors is None or right_errors is None:
        return None

    total_errors = left_errors + right_errors
    return sum(total_errors) / len(total_errors)


def calculate_error_percentage(true_left, true_right, found_left, found_right):
    if not validate_inputs(true_left, true_right, found_left, found_right):
        return None

    left_errors = calculate_single_boundary_errors(true_left, found_left)
    right_errors = calculate_single_boundary_errors(true_right, found_right)

    return calculate_average_error(left_errors, right_errors)


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


def calculate_boundary_errors(true_left, true_right, found_left, found_right):
    if len(true_left) != len(found_left) or len(true_right) != len(found_right):
        raise ValueError("Количество границ не совпадает!")

    left_errors = np.abs(np.array(true_left) - np.array(found_left))
    right_errors = np.abs(np.array(true_right) - np.array(found_right))

    # Суммарное отклонение (MAE)
    total_error = np.sum(left_errors) + np.sum(right_errors)

    individual_errors = {
        'left_errors': left_errors.tolist(),
        'right_errors': right_errors.tolist()
    }

    return total_error, individual_errors


if __name__ == "__main__":
    boundary_dict = {'left': [0, 150, 300],
                     'right': [100, 250, 400]}
    Pe = [3000, 1000, 0]
    N = 200
    sigma = 0.0001
    TG0 = 1
    atg = 0.0001
    A = 5

    # Получаем границы и данные для визуализации
    found_left, found_right, z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices = get_boundaries(
        boundary_dict, Pe, N, sigma, TG0, atg, A)

    # Визуализация нормированных данных
    plot_data(z_norm, T_true_norm, T_noisy_norm, T_smooth, start_indices, end_indices)

    # Вычисляем процентную погрешность
    error_percentage = calculate_error_percentage(boundary_dict['left'], boundary_dict['right'], found_left,
                                                  found_right)

    # Вычисляем MAE и MSE
    total_error_mae, individual_errors = calculate_boundary_errors(
        boundary_dict['left'], boundary_dict['right'], found_left, found_right
    )

    # Вычисляем MSE отдельно
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