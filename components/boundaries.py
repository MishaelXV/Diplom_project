def extract_boundaries(boundary_values):
    left_boundary = boundary_values.get('left', [])
    right_boundary = boundary_values.get('right', [])
    left_boundary = [float(b) for b in left_boundary]
    right_boundary = [float(b) for b in right_boundary]
    return left_boundary, right_boundary