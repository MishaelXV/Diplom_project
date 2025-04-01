import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from calculates_block.calculates import calculate_TsGLin_array, TsGLin
import seaborn as sns

# Определение общих констант
zInf = 100000
TG0 = 1
atg = 0.0001
A = 5
b = [0, 150]
c = [100, 250]
Pe = [[8000, 2000, 200], 0]

# for i in Pe[0]:
#     TsGLin_init = 0
#     TsGLin_array = pre.calculate_TsGLin_array(c, zInf, TG0, atg, A, [i, 0], b, TsGLin_init, len(c))
#     TsGLin_array = [float(value) for value in TsGLin_array]
#     print(TsGLin_array)
#
# print(f"Нулевое значение: {TsGLin_array[0]}")


def piecewise_constant(params, z, TsGLin_array):

    result = np.full_like(z, np.nan, dtype=float)

    # обрабатываем первый интервал
    result = np.where(
        (z >= b[0]) & (z < c[0]),  # Проверяем, что z в пределах первого интервала
        TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{1}', 0.0)), b[0], TsGLin_array[0]),
        result
    )

    # проходимся по всем интервалам, кроме последнего
    for i in range(len(c) - 1):
        result = np.where(
            (z >= c[i]) & (z < b[i + 1]),
            TsGLin_array[i + 1],
            result
        )

        result = np.where(
            (z >= b[i + 1]) & (z < c[i + 1]),
            TsGLin(z, zInf, TG0, atg, A, float(params.get(f'Pe_{i + 2}', 0.0)), b[i + 1], TsGLin_array[i + 1]),
            result
        )

    # обрабатываем последний интервал
    result = np.where(
        z >= c[-1],
        TsGLin_array[-1],
        result
    )

    return result

def residuals(params, x, y):
    return piecewise_constant(params, x, TsGLin_array) - y

params = Parameters()
for i in range(len(Pe) - 1):
    params.add(f'Pe_{i + 1}', value=1, min=0, max=50000)

sigma_values = [0.01, 0.05, 0.1]
N_rnd = 1000

all_data = []

for n in Pe[0]:
    TsGLin_init = 0
    TsGLin_array = calculate_TsGLin_array(c, zInf, TG0, atg, A, [n, 0], b, TsGLin_init, len(c))
    TsGLin_array = [float(value) for value in TsGLin_array]
    for sigma in sigma_values:
        for _ in range(N_rnd):
            x_data = np.linspace(b[0], c[-1], 50)
            y_data = piecewise_constant({f'Pe_{i+1}': n}, x_data, TsGLin_array) + np.random.normal(0, sigma, x_data.size)

            result = minimize(residuals, params, args=(x_data, y_data), method="leastsq")

            for i, param in enumerate(result.params.values()):
                delta = abs(param.value - n)
                all_data.append({"sigma": sigma, "delta": delta, "Pe": n})

# Создание DataFrame
df = pd.DataFrame(all_data)
print(df)

sns.set_palette("pastel")
plt.figure(figsize=(10, 6))
sns.boxplot(x="sigma", y="delta", data=df, hue = 'Pe', palette="muted", showmeans=True, meanprops={
        "marker": "X",
        "markerfacecolor": "red",
        "markeredgecolor": "black",
        "markersize": 8
    })
plt.title("Распределение отклонений параметров для каждого уровня шума")
plt.xlabel("Уровень шума σ")
plt.ylabel("Модуль отклонений Pe на 1 интервале")
plt.grid()
plt.savefig("boxplot_Pe.png", dpi = 300, format = "png")
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(
    x="sigma",
    y="delta",
    hue="Pe",
    data=df,
    palette="muted",
    scale="width"
)
plt.title("Распределение отклонений параметров для каждого уровня шума")
plt.xlabel("Уровень шума σ")
plt.ylabel("Модуль отклонений Pe на 1 интервале")
plt.grid()
plt.ylim(0, None)
plt.savefig("violinplot_Pe.png", dpi = 300, format = "png")
plt.show()

