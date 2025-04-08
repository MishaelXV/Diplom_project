import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from calculates_block.calculates import main_func
from calculates_block.data import generate_data_optim, smooth_data
from regression.optuna_search import train_models, predict_params
import pandas as pd
import os

# Параметры
Pe = [2000, 1000, 0]
zInf = 100000
TG0 = 1
atg = 0.0001
A = 5
sigma = 0.005
N = 350
b = [0, 150, 300]
c = [100, 250, 400]

z, T_noisy = generate_data_optim(b, c, Pe, zInf, TG0, atg, A, sigma, N)
T_true = main_func({f'Pe_{i + 1}': Pe[i] for i in range(len(Pe) - 1)}, z, zInf, TG0, atg, A, Pe, b, c)

z_min, z_max = z.min(), z.max()
T_min, T_max = T_noisy.min(), T_noisy.max()

z_norm = (z - z_min) / (z_max - z_min)
T_noisy_norm = (T_noisy - T_min) / (T_max - T_min)
T_true_norm = (T_true - T_min) / (T_max - T_min)
T_smooth_norm = smooth_data(T_noisy_norm)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RELATIVE_PATH = os.path.join('regression', 'training_data.txt')

FULL_PATH = os.path.join(BASE_DIR, RELATIVE_PATH)

df = pd.read_csv(FULL_PATH, sep='\t')

model_ws, model_ms = train_models(df)
window_size, min_slope = predict_params(Pe[0], A, sigma, N, model_ws, model_ms)

fig, ax = plt.subplots(figsize=(12, 6))
line_true, = ax.plot(z_norm, T_true_norm, label="Истинная кривая (норм.)", color="black", alpha=0.5)
line_smooth, = ax.plot(z_norm, T_smooth_norm, label="Сглаженная кривая (норм.)", color="red", alpha=0.7)
window_line, = ax.plot([], [], color='blue', linewidth=2, label="Линейная регрессия")
current_pos_left = ax.axvline(0, color='purple', linestyle='--', alpha=0.5, label="Границы окна")
current_pos_right = ax.axvline(0, color='purple', linestyle='--', alpha=0.5)
text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

ax.set_ylim(T_smooth_norm.min(), T_smooth_norm.max() + 0.2)
ax.set_xlim(z_norm.min(), z_norm.max())
ax.set_title("Анимация оконной линейной регрессии (нормированные данные)", fontsize=14)
ax.set_xlabel("Нормированная координата z", fontsize=12)
ax.set_ylabel("Нормированная температура T", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

highlight_regions = []


def init():
    global highlight_regions
    for patch in highlight_regions:
        patch.remove()
    highlight_regions = []
    window_line.set_data([], [])
    current_pos_left.set_xdata([0, 0])
    current_pos_right.set_xdata([0, 0])
    text_info.set_text('')
    return [window_line, current_pos_left, current_pos_right, text_info] + highlight_regions


def merge_regions(regions):
    if not regions:
        return []
    regions = sorted(regions, key=lambda r: r.get_x())
    merged = [regions[0]]
    for current in regions[1:]:
        last = merged[-1]
        if current.get_x() <= last.get_x() + last.get_width():
            new_x = last.get_x()
            new_width = max(last.get_x() + last.get_width(),
                            current.get_x() + current.get_width()) - new_x
            last.set_width(new_width)
        else:
            merged.append(current)
    return merged


def update(i):
    global highlight_regions

    start = i
    end = i + window_size
    if end >= len(z_norm):
        return [window_line, current_pos_left, current_pos_right, text_info] + highlight_regions

    z_window = z_norm[start:end].reshape(-1, 1)
    T_window = T_smooth_norm[start:end]
    model = LinearRegression().fit(z_window, T_window)
    T_pred = model.predict(z_window)
    slope = model.coef_[0]
    intercept = model.intercept_

    window_line.set_data(z_window.flatten(), T_pred)

    current_pos_left.set_xdata([z_norm[start], z_norm[start]])
    current_pos_right.set_xdata([z_norm[end], z_norm[end]])

    if slope > min_slope:
        new_rect = Rectangle((z_norm[start], T_smooth_norm.min() - 0.15),
                             z_norm[end] - z_norm[start],
                             T_smooth_norm.max() - T_smooth_norm.min() + 0.3,
                             color='green', alpha=0.15)

        for patch in highlight_regions:
            patch.remove()

        temp_regions = highlight_regions.copy()
        temp_regions.append(new_rect)

        merged_regions = merge_regions(temp_regions)

        highlight_regions = []
        for region in merged_regions:
            new_patch = Rectangle((region.get_x(), region.get_y()),
                                  region.get_width(), region.get_height(),
                                  color='green', alpha=0.15)
            ax.add_patch(new_patch)
            highlight_regions.append(new_patch)

    info_text = (f"Позиция: {i}/{len(z_norm)}\n"
                 f"Окно: [{z_norm[start]:.3f}, {z_norm[end]:.3f}]\n"
                 f"Наклон: {slope:.6f}\n"
                 f"Порог: {min_slope:.6f}\n"
                 f"Условие роста: {'ДА' if slope > min_slope else 'нет'}")
    text_info.set_text(info_text)

    return [window_line, current_pos_left, current_pos_right, text_info] + highlight_regions


ani = FuncAnimation(fig, update, frames=range(len(z_norm) - window_size + 1),
                    init_func=init, blit=True, interval=100, repeat=True)

plt.tight_layout()
plt.show()

# Для сохранения в файл
ani.save('norm_animation.mp4', writer='ffmpeg', fps=15, dpi=200)