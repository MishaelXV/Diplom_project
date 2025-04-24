import matplotlib.pyplot as plt
from calculates_block.main_functions import main_func, reconstruct_Pe_list
from main_algorithm.constants import COMMON_CONSTANTS

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

def plot_comparison(x_data, y_data, params, found_left, found_right):
    Pe = reconstruct_Pe_list(params, COMMON_CONSTANTS['Pe'][0])
    plt.figure(figsize=(10, 6))

    plt.plot(x_data, y_data, 'b-', label='Профиль температуры', linewidth=2)

    y_model = main_func(x_data, COMMON_CONSTANTS['TG0'], COMMON_CONSTANTS['atg'], COMMON_CONSTANTS['A'], Pe, found_left, found_right)
    plt.plot(x_data, y_model, 'g-', label='Модельный профиль', linewidth=2.5, alpha=0.8)

    plt.fill_between(x_data, y_data, y_model, color='red', alpha=0.3, label='Отклонение')

    title = 'Сравнение профилей'
    plt.title(title)
    plt.xlabel('z/rw')
    plt.ylabel('θ')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()