import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from block.block import TsGLin, calculate_TsGLin_array
from sklearn.linear_model import LinearRegression

# Исходные данные
left_boundary = [100, 250, 400, 550]
right_boundary = [200, 350, 500, 650]
TG0 = 1
atg = 0.0001
A = 10
Pe = [2000, 1500, 200, 0]
a = len(Pe)
N = 40
sigma = 0.001

TsGLin_init = 0
TsGLin_array = calculate_TsGLin_array(right_boundary, 100000, TG0, atg, A, Pe, left_boundary, TsGLin_init)
TsGLin_array = [float(value) for value in TsGLin_array]

z_all = []
T_all = []

for j in range(a):
    z = np.linspace(left_boundary[j], right_boundary[j], N)
    T_list = [TsGLin([z_val], 100000, TG0, atg, A, Pe[j], left_boundary[j], TsGLin_array[j]) for z_val in z]
    T = [item[0] for item in T_list]

    z_all.extend(z)
    T_all.extend(T)

    if j < a - 1:
        y_value = TsGLin_array[j + 1]
        start_point = right_boundary[j] + 1e-6
        end_point = left_boundary[j + 1]
        z_horizontal = np.linspace(start_point, end_point, N)
        T_all.extend([y_value] * len(z_horizontal))
        z_all.extend(z_horizontal)

noise = np.random.normal(0, sigma, size=len(T_all))
T_noisy = np.array(T_all) + noise

T_smooth = medfilt(T_noisy, kernel_size=57)

filtered_intervals = np.zeros_like(T_smooth, dtype=bool)

if sigma == 0:
    dTdz = np.gradient(T_smooth, z_all)
    filtered_intervals = dTdz > 1e-10
else:
    window_size = 5
    min_slope = 0.0001

    for i in range(len(z_all) - window_size):
        z_window = np.array(z_all[i:i + window_size]).reshape(-1, 1)
        T_window = np.array(T_smooth[i:i + window_size])

        model = LinearRegression().fit(z_window, T_window)
        slope = model.coef_[0]

        if slope > min_slope:
            filtered_intervals[i:i + window_size] = True

plt.figure(figsize=(10, 5))
plt.plot(z_all, T_all, label="Исходная кривая")
plt.plot(z_all, T_noisy, label="Шумная кривая", alpha=0.5)
plt.plot(z_all, T_smooth, label="Сглаженная кривая", linewidth=2, color='red')

start_indices = []
end_indices = []
in_interval = False

for i in range(1, len(z_all)):
    if filtered_intervals[i] and not in_interval:
        start_indices.append(i - 1)  # Индекс начала интервала
        in_interval = True
    elif not filtered_intervals[i] and in_interval:
        end_indices.append(i - 1)  # Индекс конца интервала
        in_interval = False

if end_indices and end_indices[-1] < len(z_all) - 1:
    end_indices[-1] = len(z_all) - 1

for start, end in zip(start_indices, end_indices):
    z_interval = z_all[start:end + 1]
    T_interval = T_smooth[start:end + 1]

for start, end in zip(start_indices, end_indices):
    print(f"Начало интервала: z = {z_all[start]:.2f}, T = {T_smooth[start]:.2f}")
    print(f"Конец интервала: z = {z_all[end]:.2f}, T = {T_smooth[end]:.2f}")
    print("-" * 30)


    plt.axvspan(z_all[start], z_all[end], color='green', alpha=0.2)

plt.xlabel("z")
plt.ylabel("T")
plt.title("Сглаживание")
plt.legend()
plt.show()