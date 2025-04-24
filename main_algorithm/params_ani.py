import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

def plot_animated_Pe(df_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    pe_columns = [col for col in df_history.columns if col.startswith('Pe_')]

    ax.set_xlim(-0.5, len(df_history) + 0.5)
    y_min = df_history[pe_columns].min().min() * 0.8
    y_max = df_history[pe_columns].max().max() * 1.2
    ax.set_ylim(y_min, y_max)

    ax.set_title('Изменение Pe по итерациям')
    ax.set_xlabel('Номер итерации')
    ax.set_ylabel('Значение Pe')
    ax.grid(True, linestyle='--', alpha=0.6)

    lines = []
    for i, col in enumerate(pe_columns):
        line, = ax.plot([], [], '-', linewidth=2, label=col)
        lines.append(line)

    true_lines = []

    for i, (col, true_val) in enumerate(zip(pe_columns, COMMON_CONSTANTS['Pe'][1:-1])):
        hline = ax.axhline(true_val, linestyle='--', alpha=0.7,
                           label=f'Истинное {col} = {true_val}')
        true_lines.append(hline)

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(frameon=False)

    def init():
        for line in lines:
            line.set_data([], [])
        text.set_text('')
        return lines + [text]

    def update(frame):
        x = np.arange(frame + 1)

        for i, col in enumerate(pe_columns):
            y = df_history[col].values[:frame + 1]
            lines[i].set_data(x, y)

        return lines + [text]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df_history),
        init_func=init,
        blit=True,
        interval=200,
        repeat_delay=1000
    )

    plt.tight_layout()
    plt.show()
    return ani