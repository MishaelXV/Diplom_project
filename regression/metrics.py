import numpy as np
from components.valid_inputs_of_params import validate_inputs_2

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
    if not validate_inputs_2(true_left, true_right, found_left, found_right):
        return None

    left_errors = calculate_single_boundary_errors(true_left, found_left)
    right_errors = calculate_single_boundary_errors(true_right, found_right)

    return calculate_average_error(left_errors, right_errors)


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
