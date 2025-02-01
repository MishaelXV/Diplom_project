from lmfit import minimize, Parameters
import numpy as np
from block import bb, preprocessing as pre
import pandas as pd

rng = np.random.default_rng(42)

def piecewise_constant(params, z, zInf, TG0, atg, A, Pe, b, c):
    """
    Вычисляет кусочно-постоянную функцию температуры вдоль глубины.

    Параметры
    ----------
    params : lmfit.Parameters
        Параметры оптимизации, включающие Pe для каждого интервала.
    z : array_like
        Глубины, на которых рассчитывается температура.
    zInf : float
        Бесконечная глубина, на которой достигается равновесная температура.
    TG0 : float
        Начальная температура на поверхности.
    atg : float
        Геотермический градиент.
    A : float
        Параметр, определяющий масштаб температурного распределения.
    Pe : list of float
        Расходы (параметры) для каждого интервала.
    b : list of float
        Границы изолированных участков скважины (начало интервала).
    c : list of float
        Границы изолированных участков скважины (конец интервала).

    Возвращает
    -------
    result : numpy.ndarray
        Значения температуры на глубинах, заданных в `z`.
    """
    # Получаем значения температуры
    TsGLin_init = 0
    TsGLin_array = pre.calculate_TsGLin_array(c, zInf, TG0, atg, A, Pe, b, TsGLin_init, len(c))
    TsGLin_array = [float(value) for value in TsGLin_array]

    result = np.full_like(z, np.nan, dtype=float)

    # Обрабатываем первый интервал
    result = np.where(
        (z >= b[0]) & (z < c[0]),  # Проверяем, что z в пределах первого интервала
        bb.TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{1}', 0.0)), b[0], TsGLin_array[0]),
        result
    )

    for i in range(len(c) - 1):
        result = np.where(
            (z >= c[i]) & (z < b[i + 1]),
            TsGLin_array[i + 1],
            result
        )

        result = np.where(
            (z >= b[i + 1]) & (z < c[i + 1]),
            bb.TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{i + 2}', 0.0)), b[i + 1], TsGLin_array[i + 1]),
            result
        )

    result = np.where(z >= c[-1], TsGLin_array[-1], result)

    return result


def residuals_(params, x, y, zInf, TG0, atg, A, Pe, b, c):
    """
   Функция остаточной ошибки для lmfit, вычисляет разницу между предсказанными и реальными значениями температуры.

   Параметры
   ----------
   params : lmfit.Parameters
       Параметры, которые оптимизируются.
   x : array_like
       Массив глубин, на которых осуществляется расчет.
   y : array_like
       Реальные значения температуры, с которыми проводится сравнение.
   zInf : float
       Бесконечная глубина, на которой температура становится равновесной.
   TG0 : float
       Начальная температура на поверхности.
   atg : float
       Геотермический градиент.
   A : float
       Масштабный параметр, влияющий на распределение температуры.
   Pe : list of float
       Параметры, определяющие теплообмен в каждом интервале.
   b : list of float
       Границы начала интервалов.
   c : list of float
       Границы конца интервалов.

   Возвращает
   -------
   residuals : numpy.ndarray
       Остатки (разница) между предсказанными значениями температуры и реальными данными.
   """
    return piecewise_constant(params, x, zInf, TG0, atg, A, Pe, b, c) - y

def run_optimization(b, c, Pe, zInf, TG0, atg, A, sigma, N, rng):
    """
    Выполняет оптимизацию параметров для кусочно-постоянной функции температуры.

    Параметры
    ----------
    b : list of float
        Границы изолированных участков скважины (начало интервала).
    c : list of float
        Границы изолированных участков скважины (конец интервала).
    Pe : list of float
        Расходы (параметры) для каждого интервала.
    zInf : float
        Бесконечная глубина, на которой достигается равновесная температура.
    TG0 : float
        Начальная температура на поверхности.
    atg : float
        Геотермический градиент.
    A : float
        Параметр, определяющий масштаб температурного распределения.
    sigma : float
        Стандартное отклонение шума в данных.
    seed: int, optional
        Значение для инициализации генератора случайных чисел. Если None, используется текущее значение.

    Возвращает
    -------
    param_history : list of tuple
        История изменения параметров и невязки на каждой итерации.
    df_history : pandas.DataFrame
        Таблица с историей параметров и значениями невязки.
    x_data : numpy.ndarray
        Глубины, использованные для генерации данных.
    y_data : numpy.ndarray
        Сгенерированные данные температуры с добавлением шума.
    """


    # Остаточная функция для lmfit
    def residuals(params, x, y):
        """
        Функция остаточной ошибки для lmfit, вычисляет разницу между предсказанными значениями и реальными значениями.

        Параметры
        ----------
        params : lmfit.Parameters
            Параметры, которые оптимизируются, включая значения для Pe.
        x : array_like
            Массив глубин, на которых рассчитывается температура.
        y : array_like
            Массив реальных значений температуры, с которыми проводится сравнение.

        Возвращает
        -------
        residuals : numpy.ndarray
            Остатки (разница) между предсказанными значениями температуры и реальными данными.
        """
        return piecewise_constant(params, x, zInf, TG0, atg, A, Pe, b, c) - y

    x_data = np.linspace(b[0], c[-1], N)
    y_data = piecewise_constant({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, x_data, zInf, TG0, atg, A, Pe, b, c) + rng.normal(0, sigma, x_data.size)

    # Настройка параметров для lmfit
    params = Parameters()
    for i in range(len(Pe) - 1):
        params.add(f'Pe_{i + 1}', value=1, min=0, max=30000)

    param_history = []


    def iter_callback(params, iter, resid, *args, **kwargs):
        """
        Функция обратного вызова для отслеживания прогресса оптимизации. Она вызывается на каждой итерации и сохраняет текущие значения параметров и невязки.

        Параметры
        ----------
        params : lmfit.Parameters
            Текущие значения оптимизируемых параметров.
        iter : int
            Номер текущей итерации.
        resid : numpy.ndarray
            Остатки (разница между предсказанными и реальными значениями).
        *args, **kwargs : дополнительные аргументы
            Дополнительные параметры, передаваемые функции обратного вызова.

        Возвращает
        -------
        None
            Функция не возвращает значений, но сохраняет данные в глобальной переменной `param_history`.
        """
        param_values = {param.name: float(param.value) for param in params.values()}
        chi_squared = float(np.sum(resid ** 2))
        param_history.append((param_values, chi_squared))


    result = minimize(residuals, params, args=(x_data, y_data), method='leastsq', iter_cb=iter_callback, ftol=1e-4, xtol = 1e-4)

    df_history = pd.DataFrame(param_history, columns=['parameters', 'Невязка'])
    df_params = pd.json_normalize(df_history['parameters'])
    df_history = pd.concat([df_history, df_params], axis=1)
    df_history = df_history.drop('parameters', axis=1)

    return result, param_history, df_history, x_data, y_data