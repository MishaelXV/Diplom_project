from callbacks.boundaries import extract_boundaries

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