import numpy as np
import matplotlib.pyplot as plt

# Создаем сетку точек
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Вычисляем значения функции
Z = np.sin(X**2 + Y**2)

# Создаем фигуру и оси
plt.figure(figsize=(10, 6))

# Рисуем изолинии
contour = plt.contour(X, Y, Z, levels=10, cmap='viridis')  # Контуры
plt.clabel(contour, inline=True, fontsize=8)  # Подписи для изолиний

# Заполненные изолинии
plt.contourf(X, Y, Z, levels=10, cmap='viridis', alpha=0.5)

# Настройка графика
plt.title('Изолинии функции $z = \\sin(x^2 + y^2)$', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.colorbar(label='z = $\\sin(x^2 + y^2)$')  # Цветовая шкала
plt.grid()
plt.show()
