import mpmath
import numpy as np

def TsGLin(z, zInf, TG0, atg, A, Pe, zl, Tl):
    """
    Расчет температуры вдоль скважины с учетом теплопередачи и параметра Пекле.

    Параметры
    ----------
    z : array_like
        Глубины, на которых рассчитывается температура.
    zInf : float
        Бесконечная глубина, на которой температура достигает равновесного значения.
    TG0 : float
        Начальная температура на поверхности.
    atg : float
        Геотермический градиент (изменение температуры с глубиной).
    A : float
        Коэффициент теплопередачи.
    Pe : float
        Параметр Пекле, характеризующий конвекцию тепла.
    zl : float
        Глубина на левой границе текущего интервала.
    Tl : float
        Температура на левой границе текущего интервала.

    Возвращает
    -------
    results : list of float
        Значения температуры для каждого значения глубины `z`.
    """
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


def calculate_TsGLin_array(c, zinf, TG0, atg, A, Pe, b, TsGLin_init, n):
    """
    Расчет значений температуры на левых границах интервалов скважины.

    Параметры
    ----------
    c : list of float
        Координаты концов изолированных участков скважины.
    zinf : float
        Бесконечная глубина, на которой температура достигает равновесного значения.
    TG0 : float
        Начальная температура на поверхности.
    atg : float
        Геотермический градиент (изменение температуры с глубиной).
    A : float
        Коэффициент теплопередачи.
    Pe : list of float
        Параметры Пекле для каждого интервала.
    b : list of float
        Координаты начала изолированных участков скважины.
    TsGLin_init : float
        Начальная температура на левой границе.
    n : int
        Количество интервалов.

    Возвращает
    -------
    TsGLin_array : list of float
        Значения температуры на левых границах каждого интервала.
    """
    TsGLin_array = [TsGLin_init]
    for i in range(n):
        TsGLin_current = TsGLin([c[i]], zinf, TG0, atg, A, Pe[i], b[i], TsGLin_array[i])
        TsGLin_array.append(TsGLin_current[0])
    return TsGLin_array


def geoterma(z, TG0, atg):
    """
    Расчет значения температуры вдоль глубины по геотермическому градиенту.

    Параметры
    ----------
    z : float or array_like
        Глубина.
    TG0 : float
        Начальная температура на поверхности.
    atg : float
        Геотермический градиент (изменение температуры с глубиной).

    Возвращает
    -------
    float or numpy.ndarray
        Значения температуры вдоль глубины.
    """
    return atg * z + TG0


def debit(Pe, lw=0.6, rw=0.1, cw=4200, row=1000):
    """
    Расчет расхода жидкости в скважине на основе параметра Пекле.

    Параметры
    ----------
    Pe : float
        Параметр Пекле, характеризующий конвекцию тепла.
    lw : float, optional, default=0.6
        Теплопроводность стенок скважины (Вт/м·К).
    rw : float, optional, default=0.1
        Внутренний радиус скважины (м).
    cw : float, optional, default=4200
        Удельная теплоемкость жидкости (Дж/кг·К).
    row : float, optional, default=1000
        Плотность жидкости (кг/м³).

    Возвращает
    -------
    float
        Значение расхода (м³/сутки).
    """
    return 24 * 3600 * (Pe * lw * np.pi * rw) / (cw * row)