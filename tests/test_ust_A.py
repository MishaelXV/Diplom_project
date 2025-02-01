import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from block.block import bb, preprocessing as pre
import seaborn as sns

# Определение общих констант
zInf = 100000
TG0 = 1
atg = 0.0001
A = [5, 3, 1]
b = [0, 150, 300]
c = [100, 250, 400]
Pe = [2000, 1000, 0]

# Получаем значения температуры
# for i in A:
#     TsGLin_init = 0
#     TsGLin_array = pre.calculate_TsGLin_array(c, zInf, TG0, atg, i, Pe, b, TsGLin_init, len(c))
#     TsGLin_array = [float(value) for value in TsGLin_array]

# Обновленная функция piecewise_constant
def piecewise_constant(params, z, TsGLin_array):
    """
    Кусочно-постоянная функция с параметрами в виде словаря.
    Адаптирована для работы с любым количеством параметров Pe.
    """

    result = np.full_like(z, np.nan, dtype=float)  # инициализируем массив с NaN, для избежания неопределенных значений

    # обрабатываем первый интервал
    result = np.where(
        (z >= b[0]) & (z < c[0]),  # Проверяем, что z в пределах первого интервала
        bb.TsGLin(z, zInf, TG0, atg, n, float(params.get(f'Pe_{1}', 0.0)), b[0], TsGLin_array[0]),
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
            bb.TsGLin(z, zInf, TG0, atg, n, float(params.get(f'Pe_{i + 2}', 0.0)), b[i + 1], TsGLin_array[i + 1]),
            result
        )

    # обрабатываем последний интервал
    result = np.where(
        z >= c[-1],
        TsGLin_array[-1],
        result
    )

    return result


# Остаточная функция для lmfit
def residuals(params, x, y):
    """
    Функция невязки для оптимизации.
    """
    return piecewise_constant(params, x, TsGLin_array) - y


params = Parameters()
for i in range(len(Pe) - 1):
    params.add(f'Pe_{i + 1}', value=1, min=0, max=5000)

sigma_values = [0.01, 0.05, 0.1]
N_rnd = 1000

results = []
mean_differences = []
all_data = []

for n in A:
    disp_results = []
    mean_results = []
    TsGLin_init = 0
    TsGLin_array = pre.calculate_TsGLin_array(c, zInf, TG0, atg, n, Pe, b, TsGLin_init, len(c))
    TsGLin_array = [float(value) for value in TsGLin_array]
    for sigma in sigma_values:
        deviations = []
        for _ in range(N_rnd):
            x_data = np.linspace(b[0], c[-1], 50)
            y_data = piecewise_constant({f'Pe_{i+1}': Pe[i] for i in range(len(Pe)-1)}, x_data, TsGLin_array) + np.random.normal(0, sigma, x_data.size)

            result = minimize(residuals, params, args=(x_data, y_data), method='leastsq')
            # print(result.params.values())

            for i, param in enumerate(result.params.values()):
                delta = abs(param.value - Pe[i])
                all_data.append({"sigma": sigma, "delta": delta, "A": n})

        disp_results.append(np.std(deviations))
        mean_results.append(np.mean(deviations))

    results.append(disp_results)
    mean_differences.append(mean_results)

# Создание DataFrame
df = pd.DataFrame(all_data)
df['sigma'] = df['sigma'].round(4)


# Визуализация
plt.figure(figsize=(10, 6))
sns.boxplot(x="sigma", y="delta", data=df, hue = 'A', palette="muted", showmeans=True, meanprops={
        "marker": "X",
        "markerfacecolor": "red",
        "markeredgecolor": "black",
        "markersize": 8
    })
plt.title("Распределение отклонений параметров для каждого уровня шума")
plt.xlabel("Уровень шума σ")
plt.ylabel("Модуль отклонений Pe")
plt.grid()
plt.savefig("boxplot_A.png", dpi = 300, format = "png")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(
    x="sigma",
    y="delta",
    hue="A",
    data=df,
    palette="muted",
    scale="width",
    inner = "box"
)
plt.title("Распределение отклонений параметров для каждого уровня шума")
plt.xlabel("Уровень шума σ")
plt.ylabel("Модуль отклонений Pe")
plt.grid()
plt.ylim(0, None)
plt.savefig("violinplot_A.png", dpi = 300, format = "png")
plt.show()


# for i, n in enumerate(N_samples):
#     sns.boxplot(x = sigma_values, y = deltas, label=f'Число замеров={n}')

# plt.xlabel('Уровень шума (σ)')
# plt.ylabel('Стандартное отклонение (σ_d)')
# plt.title('Зависимость стандартного отклонения от уровня шума')
# plt.legend()
# plt.grid()
# plt.show()

# Визуализация матожидания разностей
# for i, n in enumerate(N_samples):
#     plt.plot(sigma_values, mean_differences[i], label=f'Число замеров={n}')
#
# plt.xlabel('Уровень шума (σ)')
# plt.ylabel('Матожидание разности (E[ΔPe])')
# plt.title('Зависимость матожидания разности от уровня шума')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Проверка распределения разностей
# for i, n in enumerate(N_samples):
#     plt.figure(figsize=(8, 5))
#     sns.histplot(all_differences[i], kde=True, bins=30, label=f'Число замеров={n}')
#     plt.xlabel('Разность (ΔPe)')
#     plt.ylabel('Частота')
#     plt.title(f'Гистограмма распределения разностей (Число замеров={n})')
#     plt.legend()
#     plt.grid()
#     plt.show()