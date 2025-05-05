import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main_block.data import smooth_data

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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12})

plt.figure(figsize=(10, 5))
plt.plot(df["z/rw"], df["θ_smoothed"], label="Заданный профиль", color="red")
plt.xlabel("z/rw")
plt.ylabel("θ")
plt.title("Температурный профиль")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Models_test.png", format="png", dpi=300)
plt.close()

