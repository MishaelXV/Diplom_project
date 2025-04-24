import pandas as pd
import matplotlib.pyplot as plt
from calculates_block.data import smooth_data

df = pd.read_excel("/Users/macbookmike_1/Downloads/Real_Example/w0401.xlsx")

df['z/rw'] = df['d']/0.1
df['θ'] = (df['T (PLT)'] - 26)/(63.57778 - 26)
df['T_smoothed'] = smooth_data(df['θ'])

print(df)

plt.figure(figsize=(10, 5))
plt.plot(df["z/rw"], df["θ"], label="θ", color="blue")
plt.plot(df["z/rw"], df["T_smoothed"], label="θ_smoothed", color="red")
plt.xlabel("z")
plt.ylabel("T")
plt.title("Зависимость T от z")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


