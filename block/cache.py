from block.calculates import perform_optimization
from components.boundaries import extract_boundaries
from optimizator.intervals import get_boundaries

_cached_boundaries = None
_cached_optimization = None
_cached_boundaries_extracted = None


def get_boundaries_cached(boundary_dict, Pe, N, sigma, TG0, atg, A):
    global _cached_boundaries
    if _cached_boundaries is None:  # Если уже вычислено, просто возвращаем сохранённый результат
        _cached_boundaries = get_boundaries(boundary_dict, Pe, N, sigma, TG0, atg, A)
    return _cached_boundaries


def perform_optimization_cached(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N, force_update=False):
    global _cached_optimization
    if _cached_optimization is None or force_update:
        _cached_optimization = perform_optimization(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)
    return _cached_optimization



def extract_boundaries_cached(boundary_values, force_update=False):
    global _cached_boundaries_extracted
    if _cached_boundaries_extracted is None or force_update:
        _cached_boundaries_extracted = extract_boundaries(boundary_values)
    return _cached_boundaries_extracted