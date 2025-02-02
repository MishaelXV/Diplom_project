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


def calculate_TsGLin_array(c, zinf, TG0, atg, A, Pe, b, TsGLin_init):
    n = len(Pe)
    TsGLin_array = [TsGLin_init]
    for i in range(n):
        TsGLin_current = TsGLin([c[i]], zinf, TG0, atg, A, Pe[i], b[i], TsGLin_array[i])
        TsGLin_array.append(TsGLin_current[0])
    return TsGLin_array


def geoterma(z, TG0, atg):
    return atg * z + TG0


def debit(Pe, lw=0.6, rw=0.1, cw=4200, row=1000):
    return 24 * 3600 * (Pe * lw * np.pi * rw) / (cw * row)