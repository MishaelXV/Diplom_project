import mpmath

def TsGLin(z, zInf, TG0, atg, A, Pe, zl, Tl):
    z = mpmath.mpf(z)
    zInf = mpmath.mpf(zInf)
    TG0 = mpmath.mpf(TG0)
    atg = mpmath.mpf(atg)
    A = mpmath.mpf(A)
    Pe = mpmath.mpf(Pe)
    zl = mpmath.mpf(zl)
    Tl = mpmath.mpf(Tl)
    return 1 / ((-1 + mpmath.exp(mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl))) * A) * mpmath.exp(
        1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl) - Pe * (zInf + zl))
    ) * (
        atg * mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (z - zl) + Pe * (z + zl))) * Pe
        - atg * mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-z + zl) + Pe * (z + zl))) * Pe
        + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (zInf - zl) + Pe * (zInf + zl)))
        * (TG0 * A - atg * Pe + atg * A * z)
        + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-zInf + zl) + Pe * (zInf + zl)))
        * (-TG0 * A + atg * (Pe - A * z))
        + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (z - zInf) + Pe * (z + zInf)))
        * (TG0 * A - Tl * A - atg * Pe + atg * A * zl)
        + mpmath.exp(1 / 2 * (mpmath.sqrt(4 * A + Pe ** 2) * (-z + zInf) + Pe * (z + zInf)))
        * (-TG0 * A + Tl * A + atg * (Pe - A * zl))
    )


def calculate_TsGLin_array(c, zinf, TG0, atg, A, Pe, b, TsGLin_init, n):
  TsGLin_array = [TsGLin_init]
  for i in range(n):
    TsGLin_current = TsGLin(c[i], zinf, TG0, atg, A, Pe[i], b[i], TsGLin_array[i])
    TsGLin_array.append(TsGLin_current)
  return TsGLin_array


def Geoterma(z, TG0, atg):
   return atg*z + TG0


