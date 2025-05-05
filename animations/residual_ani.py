import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

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

def plot_animated_residuals(df_history):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(-0.5, len(df_history) + 0.5)
    y_min = df_history['Невязка'].min() * 0.8
    y_max = df_history['Невязка'].max() * 1.2
    ax.set_ylim(y_min, y_max)
    ax.set_yscale('log')

    ax.set_title('Изменение J по итерациям')
    ax.set_xlabel('Номер итерации')
    ax.set_ylabel('J')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True, linestyle='--', alpha=0.6)

    line, = ax.plot([], [], 'b-', linewidth=2, label='J')

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.8))

    def init():
        line.set_data([], [])
        text.set_text('')
        return line, text


    def update(frame):
        x = np.arange(frame + 1)
        y = df_history['Невязка'].values[:frame + 1]

        line.set_data(x, y)

        return line, text

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