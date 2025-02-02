import numpy as np
from lmfit import minimize, Parameters
from block import bb, preprocessing as pre
import pandas as pd

rng = np.random.default_rng(42)

# 1. Функция для расчета TsGLin_array (вынесена из main_func)
def calculate_TsGLin_array(c, zInf, TG0, atg, A, Pe, b, TsGLin_init):
    return [float(value) for value in pre.calculate_TsGLin_array(c, zInf, TG0, atg, A, Pe, b, TsGLin_init, len(c))]

# 2. Функция для расчета значений в одном интервале (вынесена из main_func)
def calculate_interval(z, zInf, TG0, atg, A, Pe_value, b_value, TsGLin_value):
    return bb.TsGLin(z, zInf, TG0, atg, A, Pe_value, b_value, TsGLin_value)

# 3. Функция main_func теперь становится более простой и читабельной
def main_func(params, z, zInf, TG0, atg, A, Pe, b, c):
    TsGLin_array = calculate_TsGLin_array(c, zInf, TG0, atg, A, Pe, b, 0)
    result = np.full_like(z, np.nan, dtype=float)

    result = np.where((z >= b[0]) & (z < c[0]), calculate_interval(z, zInf, TG0, atg, A, params['Pe_1'], b[0], TsGLin_array[0]), result)

    for i in range(len(c) - 1):
        result = np.where((z >= c[i]) & (z < b[i + 1]), TsGLin_array[i + 1], result)
        result = np.where((z >= b[i + 1]) & (z < c[i + 1]), calculate_interval(z, zInf, TG0, atg, A, params[f'Pe_{i + 2}'], b[i + 1], TsGLin_array[i + 1]), result)

    result = np.where(z >= c[-1], TsGLin_array[-1], result)
    return result



# 4. Остаточная функция (неизменна, но теперь использует более мелкие функции)
def residuals_(params, x, y, zInf, TG0, atg, A, Pe, b, c):
    return main_func(params, x, zInf, TG0, atg, A, Pe, b, c) - y


# 5. Функция для генерации данных (вынесена из run_optimization)
def generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    x_data = np.linspace(b[0], c[-1], N)
    y_data = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, x_data, zInf, TG0, atg, A, Pe, b, c) + rng.normal(0, sigma, x_data.size)
    return x_data, y_data

# 6. Функция для создания параметров (вынесена из run_optimization)
def create_parameters(Pe):
    params = Parameters()
    for i in range(len(Pe) - 1):
        params.add(f'Pe_{i + 1}', value=1, min=0, max=30000)
    return params

# 7. Функция для обработки результатов оптимизации (вынесена из run_optimization)
def process_results(param_history):
    df_history = pd.DataFrame(param_history, columns=['parameters', 'Невязка'])
    df_params = pd.json_normalize(df_history['parameters'])
    df_history = pd.concat([df_history, df_params], axis=1)
    df_history = df_history.drop('parameters', axis=1)
    return df_history

# 8. Основная функция оптимизации (теперь намного чище)
def run_optimization(b, c, Pe, zInf, TG0, atg, A, sigma, N):
    x_data, y_data = generate_data(b, c, Pe, zInf, TG0, atg, A, sigma, N)
    params = create_parameters(Pe)
    param_history = []

    def residuals(params, x, y):
        return main_func(params, x, zInf, TG0, atg, A, Pe, b, c) - y

    def iter_callback(params, resid):
        param_values = {param.name: float(param.value) for param in params.values()}
        chi_squared = float(np.sum(resid ** 2))
        param_history.append((param_values, chi_squared))

    result = minimize(residuals, params, args=(x_data, y_data), method='leastsq', iter_cb=iter_callback, ftol=1e-4, xtol=1e-4)
    df_history = process_results(param_history)
    return result, param_history, df_history, x_data, y_data
