import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lmfit import Parameters
from calculates_block.main_functions import main_func
from main_algorithm.constants import COMMON_CONSTANTS

def create_figure_and_axes(x_data, y_data):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(x_data, y_data, label='Профиль температуры', linewidth=2)
    line, = ax.plot([], [], 'g-', label='Модельная кривая', linewidth=2.5, alpha=0.8)
    fill = ax.fill_between([], [], [], color='red', alpha=0.3, label='Отклонение')

    ax.set_title('Подгонка кривой', fontsize=14, pad=15)
    ax.set_xlabel('z/rw', fontsize=12)
    ax.set_ylabel('θ', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    text_info = ax.text(
        0.84, 0.2, '', transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'),
        verticalalignment='top'
    )

    return fig, ax, line, fill, text_info


def init_animation(ax, line, text_info):
    line.set_data([], [])
    for coll in ax.collections[1:]:
        coll.remove()
    text_info.set_text('')
    return line, text_info


def update_animation(frame_idx, ax, line, fill, text_info, x_data, y_data, found_left, found_right, param_history):
    # Получаем текущую строку из DataFrame по индексу
    current_row = param_history.iloc[frame_idx]

    # Собираем параметры Pe из колонок Pe_0, Pe_1, Pe_2, ...
    pe_columns = [col for col in param_history.columns if col.startswith('Pe_')]
    current_Pe = [current_row[col] for col in sorted(pe_columns, key=lambda x: int(x.split('_')[1]))]
    print(current_Pe)
    # Собираем параметры delta из колонок delta_0, delta_1, ...
    delta_columns = [col for col in param_history.columns if col.startswith('delta_')]
    current_deltas = {col: current_row[col] for col in sorted(delta_columns, key=lambda x: int(x.split('_')[1]))}
    # Получаем метрики
    chi_sq = current_row['Невязка']
    iter_num = current_row['Итерация']

    # Вычисляем модельную кривую
    y_fit = main_func(
        x_data,
        COMMON_CONSTANTS['TG0'],
        COMMON_CONSTANTS['atg'],
        COMMON_CONSTANTS['A'],
        current_Pe,
        found_left,
        found_right
    )

    # Обновляем графики
    line.set_data(x_data, y_fit)

    # Очищаем предыдущую заливку
    for coll in ax.collections[1:]:
        coll.remove()
    fill = ax.fill_between(x_data, y_data, y_fit, color='red', alpha=0.3)

    # Обновляем заголовок
    ax.set_title(f'Подгонка кривой (Итерация {iter_num})')

    # Формируем информационный текст
    info_text = f"Итерация: {iter_num}\nНевязка: {chi_sq:.2e}\n"
    for i, pe_val in enumerate(current_Pe):
        info_text += f"Pe_{i}: {pe_val:.2f}\n"
    for name, value in current_deltas.items():
        info_text += f"{name}: {value:.2e}\n"

    text_info.set_text(info_text)

    return line, fill, text_info


def run_ani(found_left, found_right, param_history, x_data, y_data):
    fig, ax, line, fill, text_info = create_figure_and_axes(x_data, y_data)

    ani = FuncAnimation(
        fig,
        lambda frame_idx: update_animation(
            frame_idx, ax, line, fill, text_info,
            x_data, y_data, found_left, found_right, param_history
        ),
        frames=len(param_history),
        init_func=lambda: init_animation(ax, line, text_info),
        blit=True,
        interval=400,
        repeat_delay=3000
    )

    plt.tight_layout()
    plt.show()

    return ani