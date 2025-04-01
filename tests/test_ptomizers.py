import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import pandas as pd
import seaborn as sns
from calculates_block.calculates import calculate_TsGLin_array, TsGLin

# Определение общих констант
zInf = 100000
TG0 = 1
atg = 0.0001
A = 4
b = [0, 150, 300]
c = [100, 250, 400]
Pe_real = [2000, 1000, 0]

# Обновленная функция piecewise_constant
def main_func(params, z, Pe):
    TsGLin_init = 0
    TsGLin_array = calculate_TsGLin_array(c, zInf, TG0, atg, A, Pe, b, TsGLin_init, len(c))
    TsGLin_array = [float(value) for value in TsGLin_array]

    model = np.full_like(z, np.nan, dtype=float)
    model = np.where(
        (z >= b[0]) & (z <= c[0]),  # Проверяем, что z в пределах первого интервала
        TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{1}', 0)), b[0], TsGLin_array[0]),
        model
    )

    for i in range(len(c) - 1):
        model = np.where(
            (z >= c[i]) & (z <= b[i + 1]),
            TsGLin_array[i + 1],
            model
        )

        model = np.where(
            (z > b[i + 1]) & (z <= c[i + 1]),
            TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{i + 2}', 0)), b[i + 1], TsGLin_array[i + 1]),
            model
        )

    model = np.where(
        z >= c[-1],
        TsGLin_array[-1],
        model
    )

    return model

def residuals(params, x, y):
    return main_func(params, x, Pe_real) - y

z_all = []
for j in range(len(b)):
    z = np.linspace(b[j], c[j], 20)
    z_all.extend(z)
    if j < len(b) - 1:
        start_point = c[j] + 1e-6
        end_point = b[j + 1]
        z_horizontal = np.linspace(start_point, end_point, 5)
        z_all.extend(z_horizontal)

z_all = np.array(z_all)
x_data = z_all
# x_data = np.linspace(b[0], c[-1], 500)
y_data_raw = main_func({f'Pe_{i + 1}': Pe_real[i] for i in range(len(b) - 1)}, x_data, Pe_real)
y_data = y_data_raw + np.random.normal(0, 0.05, x_data.size)

params = Parameters()
for i in range(len(b) - 1):
    params.add(f'Pe_{i + 1}', value=1, min=0, max=30000)  # добавляем нужное количество параметров Pe_i

param_history = []

def iter_callback(params, iter, resid, *args, **kwargs):
    param_values = {param.name: float(param.value) for param in params.values()}
    chi_squared = float(np.sum(resid ** 2))
    param_history.append((param_values, chi_squared))

methods = ['leastsq']

plt.figure(figsize=(10, 6))


result = minimize(residuals, params, args=(x_data, y_data), method=methods, iter_cb=iter_callback)

df_history = pd.DataFrame(param_history, columns=['parameters', 'Невязка'])
df_params = pd.json_normalize(df_history['parameters'])
df_history = pd.concat([df_history, df_params], axis=1)
df_history = df_history.drop('parameters', axis=1)

for j, (params_dict, _) in enumerate(param_history):
    Pe_values = list(params_dict.values())
    Pe_values.append(0)

y_fit = main_func(params_dict, x_data, Pe_values)
# print(result.method)
# print(result.chisqr, result.nfev)
# print(df_history)

sns.lineplot(x = x_data, y = y_fit, label=f'Метод {methods[i]}', linewidth=2)
sns.lineplot(x = x_data, y = y_data_raw, label = 'Модельная кривая', linewidth = 2)
sns.scatterplot(x = x_data, y = y_data, label='Шум', alpha=0.7)
plt.title('Подгонка кусочно-постоянной функции')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid()
plt.show()

