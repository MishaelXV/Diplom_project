import matplotlib.pyplot as plt
import seaborn as sns

# Общие функции для построения графиков
def plot_std_deviation(results, variables, output_dir):
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(variables["N_samples"]):
        plt.plot(variables["sigma_values"], results[i], label=f'Число замеров={n}')
    plt.xlabel('Уровень шума (σ)')
    plt.ylabel('Стандартное отклонение точек профиля')
    plt.title('Зависимость стандартного отклонения точек профиля от уровня шума')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/std_deviation_vs_noise.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_boxplot(df, output_dir, hue, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="sigma", y="delta", data=df, hue=hue, palette="muted", showmeans=True, meanprops={
        "marker": "X",
        "markerfacecolor": "red",
        "markeredgecolor": "black",
        "markersize": 8
    })
    plt.title(title)
    plt.xlabel("Уровень шума σ")
    plt.ylabel("Модуль отклонения профиля")
    plt.grid()
    plt.savefig(f"{output_dir}/boxplot_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.close()


def plot_violinplot(df, output_dir, hue, title):
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="sigma",
        y="delta",
        hue=hue,
        data=df,
        palette="muted",
        scale="width",
        inner="box"
    )
    plt.title(title)
    plt.xlabel("Уровень шума σ")
    plt.ylabel("Модуль отклонения профиля")
    plt.grid()
    plt.ylim(0, None)
    plt.savefig(f"{output_dir}/violinplot_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.close()


def plot_mean_differences(mean_differences, variables, output_dir):
    plt.figure(figsize=(10, 6))
    for i, n in enumerate(variables["N_samples"]):
        plt.plot(variables["sigma_values"], mean_differences[i], label=f'Число замеров={n}')
    plt.xlabel('Уровень шума (σ)')
    plt.ylabel('Матожидание отклонения точек профиля')
    plt.title('Зависимость матожидания отклонения точек профиля от уровня шума')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/mean_differences.png", dpi=300, format="png", bbox_inches='tight')
    plt.close()


def plot_histograms(all_differences, N_samples, output_dir):
    for i, n in enumerate(N_samples):
        plt.figure(figsize=(8, 5))
        sns.histplot(all_differences[i], kde=True, bins=30, label=f'Число замеров={n}')
        plt.xlabel('Разность (ΔPe)')
        plt.ylabel('Частота')
        plt.title(f'Гистограмма отклонения точек профиля (Число замеров={n})')
        plt.legend()
        plt.grid()
        plt.savefig(f"{output_dir}/histogram_n_{n}.png", dpi=300, format="png", bbox_inches='tight')
        plt.close()