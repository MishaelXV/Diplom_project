import numpy as np
import mpmath

def TsGLin(z, zInf, TG0, atg, A, Pe, zl, Tl):
    z = np.asarray(z, dtype=float)

    def TsGLin_single(z_val):
        z_mpf = mpmath.mpf(z_val)
        zInf_mpf = mpmath.mpf(zInf)
        TG0_mpf = mpmath.mpf(TG0)
        atg_mpf = mpmath.mpf(atg)
        A_mpf = mpmath.mpf(A)
        Pe_mpf = mpmath.mpf(Pe)
        zl_mpf = mpmath.mpf(zl)
        Tl_mpf = mpmath.mpf(Tl)
        result = 1 / ((-1 + mpmath.exp(mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (zInf_mpf - zl_mpf))) * A_mpf) * mpmath.exp(
            1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (zInf_mpf - zl_mpf) - Pe_mpf * (zInf_mpf + zl_mpf))
        ) * (
            atg_mpf * mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (z_mpf - zl_mpf) + Pe_mpf * (z_mpf + zl_mpf))) * Pe_mpf
            - atg_mpf * mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (-z_mpf + zl_mpf) + Pe_mpf * (z_mpf + zl_mpf))) * Pe_mpf
            + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (zInf_mpf - zl_mpf) + Pe_mpf * (zInf_mpf + zl_mpf)))
            * (TG0_mpf * A_mpf - atg_mpf * Pe_mpf + atg_mpf * A_mpf * z_mpf)
            + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (-zInf_mpf + zl_mpf) + Pe_mpf * (zInf_mpf + zl_mpf)))
            * (-TG0_mpf * A_mpf + atg_mpf * (Pe_mpf - A_mpf * z_mpf))
            + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (z_mpf - zInf_mpf) + Pe_mpf * (z_mpf + zInf_mpf)))
            * (TG0_mpf * A_mpf - Tl_mpf * A_mpf - atg_mpf * Pe_mpf + atg_mpf * A_mpf * zl_mpf)
            + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A_mpf + Pe_mpf ** 2) * (-z_mpf + zInf_mpf) + Pe_mpf * (z_mpf + zInf_mpf)))
            * (-TG0_mpf * A_mpf + Tl_mpf * A_mpf + atg_mpf * (Pe_mpf - A_mpf * zl_mpf))
        )
        return float(result)

    vectorized_TsGLin = np.vectorize(TsGLin_single)
    return vectorized_TsGLin(z)


def calculate_TsGLin_array(c, zinf, TG0, atg, A, Pe, b, TsGLin_init):
  n = 3
  TsGLin_array = [TsGLin_init]
  for i in range(n):
    TsGLin_current = TsGLin(c[i], zinf, TG0, atg, A, Pe[i], b[i], TsGLin_array[i])
    TsGLin_array.append(TsGLin_current)
  return TsGLin_array


def Geoterma(z, TG0, atg):
   return atg*z + TG0