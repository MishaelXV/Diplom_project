import numpy as np

def adjust_interval_counts(true_left, found_left, found_right):
    penalty = 0
    expected_count = len(true_left)
    found_count = len(found_left)

    adjusted_left = list(found_left)
    adjusted_right = list(found_right)

    if found_count > expected_count:
        indexed_lengths = [(i, r - l) for i, (l, r) in enumerate(zip(adjusted_left, adjusted_right))]

        sorted_lengths = sorted(indexed_lengths, key=lambda x: x[1], reverse=True)

        top_indices = [x[0] for x in sorted_lengths[:expected_count]]

        top_indices_sorted = sorted(top_indices)

        adjusted_left = [found_left[i] for i in top_indices_sorted]
        adjusted_right = [found_right[i] for i in top_indices_sorted]
        penalty = 0.0

    elif found_count < expected_count:
        missing_count = expected_count - found_count
        for _ in range(missing_count):
            adjusted_left.insert(0, 0)
            adjusted_right.insert(0, 0)
        penalty = 0.0

    return adjusted_left, adjusted_right, penalty


def calculate_boundary_errors(true_left, true_right, found_left, found_right):
    adjusted_left, adjusted_right, penalty = adjust_interval_counts(
        true_left, found_left, found_right
    )

    left_errors = np.abs(np.array(true_left) - np.array(adjusted_left))
    right_errors = np.abs(np.array(true_right) - np.array(adjusted_right))

    total_error = np.sum(left_errors) + np.sum(right_errors) + penalty

    individual_errors = {
        'left_errors': left_errors.tolist(),
        'right_errors': right_errors.tolist(),
        'penalty': penalty
    }

    return total_error, individual_errors


def calculate_mae(true_left, true_right, found_left, found_right):
    total_mae, errors = calculate_boundary_errors(true_left, true_right, found_left, found_right)
    return total_mae


def calculate_mse(true_left, true_right, found_left, found_right):
    _, errors = calculate_boundary_errors(true_left, true_right, found_left, found_right)
    left = np.array(errors['left_errors'])
    right = np.array(errors['right_errors'])
    penalty = errors['penalty']

    left_with_penalty = left + penalty
    right_with_penalty = right + penalty

    return np.sum(left_with_penalty ** 2 + right_with_penalty ** 2)


def calculate_rmse(true_left, true_right, found_left, found_right):
    mse = calculate_mse(true_left, true_right, found_left, found_right)
    return np.sqrt(mse)


def calculate_relative_mae(true_left, true_right, found_left, found_right):
    mae = calculate_mae(true_left, true_right, found_left, found_right)
    total_length = np.sum(np.array(true_right) - np.array(true_left))
    return (mae / total_length) * 100 if total_length > 0 else 0