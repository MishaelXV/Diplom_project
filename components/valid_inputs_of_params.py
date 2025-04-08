from components.boundaries import extract_boundaries

def validate_inputs(a, b_values, boundary_values, A, TG0, atg, sigma):
    if any(param is None or param < 0 for param in [a, A, TG0, atg, sigma]):
        raise ValueError("Параметры должны быть заданы и неотрицательными")

    if isinstance(b_values, list):
        b_values = [float(b) for b in b_values]
    else:
        b_values = [float(b_values)]

    if any(b < 0 for b in b_values):
        raise ValueError("Значения Pe должны быть неотрицательными")

    if not boundary_values:
        raise ValueError("Границы должны быть заданы")

    left_boundary, right_boundary = extract_boundaries(boundary_values)

    if any(boundary < 0 for boundary in left_boundary + right_boundary):
        raise ValueError("Границы должны быть неотрицательными")

    if len(b_values) != a:
        raise ValueError("Количество значений b не соответствует количеству интервалов")

    return b_values, left_boundary, right_boundary


def validate_inputs_2(true_left, true_right, found_left, found_right):
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