import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from sklearn.linear_model import LinearRegression
from main_block.data import generate_data, smooth_data, data_norm, noize_data
from regression.global_models import model_ws, model_ms, predict_params

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Параметры
Pe = [200, 100, 0]
zInf = 100000
TG0 = 1
atg = 0.0001
A = 2
sigma = 0.001
N = 500
left_boundaries = [0, 150, 300]
right_boundaries = [100, 250, 350]

x_data, y_data = generate_data(left_boundaries, right_boundaries, Pe, TG0, atg, A, N)
y_data_noize = noize_data(y_data, sigma)
z_norm, T_noisy_norm = data_norm(x_data, y_data_noize)
T_smooth_norm = smooth_data(T_noisy_norm)

window_size, min_slope = predict_params(Pe[0], A, sigma, N, model_ws, model_ms)

fig, ax = plt.subplots(figsize=(12, 6))
line_smooth, = ax.plot(z_norm, T_smooth_norm, label="Температурный профиль", color="red", alpha=0.7)
window_line, = ax.plot([], [], color='blue', linewidth=2, label="Линейная регрессия")
current_pos_left = ax.axvline(0, color='purple', linestyle='--', alpha=0.5, label="Границы окна")
current_pos_right = ax.axvline(0, color='purple', linestyle='--', alpha=0.5)
text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

ax.set_ylim(T_smooth_norm.min(), T_smooth_norm.max() + 0.2)
ax.set_xlim(z_norm.min(), z_norm.max())
ax.set_title("Оконная линейная регрессия")
ax.set_xlabel("z/rw")
ax.set_ylabel("θ")
ax.legend(frameon=False)
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
        is_last_growing = True
        for j in range(i + 1, len(z_norm) - window_size + 1):
            next_z = z_norm[j:j + window_size].reshape(-1, 1)
            next_T = T_smooth_norm[j:j + window_size]
            next_model = LinearRegression().fit(next_z, next_T)
            if next_model.coef_[0] > min_slope:
                is_last_growing = False
                break

        if is_last_growing:
            rect_end = z_norm.max()
        else:
            rect_end = z_norm[end]

        new_rect = Rectangle((z_norm[start], T_smooth_norm.min() - 0.15),
                             rect_end - z_norm[start],
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
                                  color='green', alpha=0.1)
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
ani.save('regressor_animation.gif', writer='pillow', fps=15)