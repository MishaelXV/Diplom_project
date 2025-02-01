import numpy as np
import matplotlib.pyplot as plt
from block.block import TsGLin, calculate_TsGLin_array

left_boundary = [100, 250, 400]
right_boundary = [200, 350, 500]
TG0 = 1
atg = 0.0001
A = 1
Pe = [200, 100, 0]
a = len(Pe)

TsGLin_init = 0
TsGLin_array = calculate_TsGLin_array(right_boundary, 100000, TG0, atg, A, Pe, left_boundary,
                                      TsGLin_init,
                                      a)
TsGLin_array = [float(value) for value in TsGLin_array]
z_all = []
T_all = []

for j in range(a):
    z = np.linspace(left_boundary[j], right_boundary[j], 50)
    T_list = [TsGLin([z_val], 100000, TG0, atg, A, Pe[j], left_boundary[j], TsGLin_array[j]) for z_val
              in z]
    T = [item[0] for item in T_list]

    z_all.extend(z)
    T_all.extend(T)

    if j < a - 1:
        y_value = TsGLin_array[j + 1]
        start_point = right_boundary[j] + 1e-6
        end_point = left_boundary[j + 1]
        z_horizontal = np.linspace(start_point, end_point, 20)
        T_all.extend([y_value] * len(z_horizontal))
        z_all.extend(z_horizontal)

noise = np.random.normal(0, 0.01, size=len(T_all))  # добавляем шум
T = T_all + noise  # итоговая кривая с шумом

# Сглаживание для подавления шума
window_size = 5
T_smooth = np.convolve(T, np.ones(window_size)/window_size, mode='same')


# Вычисление градиента
gradient = np.gradient(T_smooth, z_all)

# Условие для определения участков роста
epsilon = 0.1
growth_zones = gradient > epsilon  # True для участков роста

# Поиск границ участков роста
boundaries = np.diff(growth_zones.astype(int))  # Находим переходы
start_indices = np.where(boundaries == 1)[0] + 1  # Начало участка роста
end_indices = np.where(boundaries == -1)[0]      # Конец участка роста

# Если участок роста доходит до конца массива
if growth_zones[-1]:
    end_indices = np.append(end_indices, len(z_all) - 1)

# Определяем границы
min_length = left_boundary[1] - right_boundary[0]  # Минимальная длина участка в точках
growth_intervals = [(z_all[start], z_all[end]) for start, end in zip(start_indices, end_indices)
                    if (end - start) >= min_length]

# Вывод результата
print("Участки роста (границы):")
for i, (start, end) in enumerate(growth_intervals):
    print(f"Интервал {i + 1}: от {start:.2f} м до {end:.2f} м")

# Визуализация
plt.figure(figsize=(10, 6))

# Оригинальная кривая с шумом
plt.plot(z_all, T, label='Модельная кривая с шумом', alpha=0.6)
# Сглаженная кривая
plt.plot(z_all, T_smooth, label='Сглаженная кривая', linewidth=2)

plt.plot(z_all, T_all, label = 'Модельная кривая', linewidth=2)
# Выделение участков роста
for start, end in growth_intervals:
    plt.axvspan(start, end, color='green', alpha=0.3, label=f'Участок роста ({start:.1f} - {end:.1f})')

plt.xlabel('z/rw', fontname='Times New Roman', fontsize=14 * 1.4)
plt.ylabel('θ', fontname='Times New Roman', fontsize=14 * 1.4)
plt.title('Определение участков роста с учетом шума', fontname='Times New Roman', fontsize=14 * 1.4)
plt.legend()
plt.show()