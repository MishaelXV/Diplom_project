import pandas as pd
import matplotlib.pyplot as plt
from lmfit import minimize
from main_block.data import smooth_data
from main_block.main_functions import reconstruct_Pe_list, main_func
from optimizator.optimizer import optimization_residuals, create_parameters
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms

df = pd.read_excel("/Users/macbookmike_1/Downloads/Real_Example/w0401.xlsx")
df = df[df.index < 2389]

df['z/rw'] = df['d']/0.1
df['θ'] = (df['T (PLT)'] - 26)/(63.57778 - 26)
df['θ_smoothed'] = smooth_data(df['θ'])

TG0 = 1
atg = 0.0001
A = 2
Pe = [16000]
N = 2400
sigma = 0.001

found_left, found_right = get_boundaries(df['z/rw'], df['θ_smoothed'], Pe, N, sigma, A, model_ws, model_ms)

params = create_parameters(found_left, Pe[0])

result = minimize(
        lambda p, x, y: optimization_residuals(p, x, y, TG0, atg, A, Pe, found_left, found_right),
        params,
        args=(df['z/rw'], df['θ']),
        method='leastsq',
        nan_policy='omit'
    )

Pe_opt = reconstruct_Pe_list(result.params, Pe[0])
y_temp = main_func(df['z/rw'], TG0, atg, A, Pe_opt, found_left, found_right)

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300
    })

plt.figure(figsize=(10, 5))
plt.plot(df["z/rw"], df["θ_smoothed"], label="Заданный профиль", color="red")
plt.plot(df['z/rw'], y_temp, label="Модельный профиль", color="blue")
plt.xlabel("z/rw")
plt.ylabel("θ")
plt.title("Температурный профиль")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Models_test.png", format="png", dpi=300)
plt.close()

