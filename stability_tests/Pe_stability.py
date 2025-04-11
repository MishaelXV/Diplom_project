import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import seaborn as sns
from calculates_block.main_functions import main_func

output_dir = "data/charts/stability_Pe"
os.makedirs(output_dir, exist_ok=True)

# Определение общих констант
zInf = 100000
TG0 = 1
atg = 0.0001
A = 5
b = [0, 150]
c = [100, 250]
Pe = [[8000, 2000, 200], 0]

def residuals(params, x, y):
    return main_func(params, x, zInf, TG0, atg, A, Pe, b, c) - y

params = Parameters()
for i in range(len(Pe) - 1):
    params.add(f'Pe_{i + 1}', value=1, min=0, max=50000)

sigma_values = [0.01, 0.05, 0.1]
N_rnd = 1

all_data = []

for n in Pe[0]:
    for sigma in sigma_values:
        for _ in range(N_rnd):
            x_data = np.linspace(b[0], c[-1], 50)
            y_data = main_func({f'Pe_{i+1}': n}, x_data, zInf, TG0, atg, A, Pe, b, c) + np.random.normal(0, sigma, x_data.size)

            result = minimize(residuals, params, args=(x_data, y_data), method="leastsq")

            for i, param in enumerate(result.params.values()):
                delta = abs(param.value - n)
                all_data.append({"sigma": sigma, "delta": delta, "Pe": n})

df = pd.DataFrame(all_data)

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
plt.savefig(f"{output_dir}/boxplot_graph.png", dpi=300, format="png", bbox_inches='tight')
plt.close()

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
plt.savefig(f"{output_dir}/violinplot_graph.png", dpi=300, format="png", bbox_inches='tight')
plt.close()

