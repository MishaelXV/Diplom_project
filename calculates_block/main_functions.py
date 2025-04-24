import mpmath
import numpy as np

def TsGLin(z, zInf, TG0, atg, A, Pe, zl, Tl):
    z = [mpmath.mpf(z_i) for z_i in z]
    zInf = mpmath.mpf(zInf)
    TG0 = mpmath.mpf(TG0)
    atg = mpmath.mpf(atg)
    A = mpmath.mpf(A)
    Pe = mpmath.mpf(Pe)
    zl = mpmath.mpf(zl)
    Tl = mpmath.mpf(Tl)

    results = []
    for z_val in z:
        result = 1 / ((-1 + mpmath.exp(mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl))) * A) * mpmath.exp(
            1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl) - Pe * (zInf + zl))
        ) * (
             atg * mpmath.exp(
         1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (z_val - zl) + Pe * (z_val + zl))) * Pe
             - atg * mpmath.exp(
         1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-z_val + zl) + Pe * (z_val + zl))) * Pe
             + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl) + Pe * (zInf + zl)))
             * (TG0 * A - atg * Pe + atg * A * z_val)
             + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-zInf + zl) + Pe * (zInf + zl)))
             * (-TG0 * A + atg * (Pe - A * z_val))
             + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (z_val - zInf) + Pe * (z_val + zInf)))
             * (TG0 * A - Tl * A - atg * Pe + atg * A * zl)
             + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-z_val + zInf) + Pe * (z_val + zInf)))
             * (-TG0 * A + Tl * A + atg * (Pe - A * zl))
                 )
        results.append(float(result))
    return results


def calculate_TsGLin_array(c, TG0, atg, A, Pe, b, TsGLin_init):
    n = len(Pe)
    TsGLin_array = [TsGLin_init]
    for i in range(n):
        TsGLin_current = TsGLin([c[i]], 1000000, TG0, atg, A, Pe[i], b[i], TsGLin_array[i])
        TsGLin_array.append(TsGLin_current[0])
    return TsGLin_array


def reconstruct_Pe_list(params, Pe_1):
    deltas = [float(p.value) for name, p in params.items() if name.startswith('delta')]
    Pe_values = [Pe_1]
    for d in deltas:
        Pe_values.append(Pe_values[-1] - d)
    return Pe_values


def main_func(z, TG0, atg, A, Pe, left_boundaries, right_boundaries):
    TsGLin_array = calculate_TsGLin_array(right_boundaries, TG0, atg, A, Pe, left_boundaries, 0)
    result = np.full_like(z, np.nan, dtype=float)
    result = np.where(
        (z >= left_boundaries[0]) & (z < right_boundaries[0]),
        TsGLin(z, 1000000, TG0, atg, A, Pe[0], left_boundaries[0], TsGLin_array[0]),
        result
    )

    for i in range(len(right_boundaries) - 1):
        result = np.where(
            (z >= right_boundaries[i]) & (z < left_boundaries[i + 1]),
            TsGLin_array[i + 1],
            result
        )
        result = np.where(
            (z >= left_boundaries[i + 1]) & (z < right_boundaries[i + 1]),
            TsGLin(z, 100000, TG0, atg, A, Pe[i + 1], left_boundaries[i + 1], TsGLin_array[i + 1]),
            result
        )

    result = np.where(z >= right_boundaries[-1], TsGLin_array[-1], result)
    return result


def geoterma(z, TG0, atg):
    return atg * z + TG0