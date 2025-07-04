import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300
    })

def plot_histograms(df, N_samples, output_dir):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", n_colors=len(N_samples))

    for idx, n in enumerate(N_samples):
        subset = df[df["N_samples"] == n]
        plt.hist(subset["deviation_metric"], bins=30,
                 alpha=0.5, color=palette[idx],
                 label=f'N={n}', edgecolor='black', linewidth=0.8)

    plt.xlabel('J')
    plt.ylabel('Значения')
    plt.title('Распределение J')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histogram_combined.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/histogram_combined.pdf", bbox_inches='tight')
    plt.close()

    for n in N_samples:
        plt.figure(figsize=(10, 6))
        subset = df[df["N_samples"] == n]
        sns.histplot(subset["deviation_metric"], bins=30,
                     color='skyblue', edgecolor='black', linewidth=1.2)
        sns.kdeplot(
            subset["deviation_metric"],
            color='darkblue',
            linewidth=2)
        plt.xlabel('J')
        plt.ylabel('Значения')
        plt.title('Распределение невязки решения')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/histogram_n_{n}.png", bbox_inches='tight')
        plt.savefig(f"{output_dir}/histogram_n_{n}.pdf", bbox_inches='tight')
        plt.close()


def plot_mean_differences(df, variables, output_dir):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    for n in variables["N_samples"]:
        subset = df[df["N_samples"] == n]
        means = subset.groupby("sigma")["deviation_metric"].mean()
        plt.plot(variables["sigma_values"], means,
                label=f'N={n}', linewidth=2, markersize=5)
    plt.xlabel('Уровень шума')
    plt.ylabel('Матожидание')
    plt.title('Матожидание J от уровня шума для разного числа замеров')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_deviation.png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/mean_deviation.pdf", bbox_inches='tight')
    plt.close()


def plot_std_deviation(df, variables, output_dir):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    for n in variables["N_samples"]:
        stds = []
        for sigma in variables["sigma_values"]:
            subset = df[(df["N_samples"] == n) & (df["sigma"] == sigma)]
            std = subset["deviation_metric"].std()
            stds.append(std)

        plt.plot(variables["sigma_values"], stds,
                 label=f'N={n}', linewidth=2, markersize=5)

    plt.xlabel('Уровень шума')
    plt.ylabel('Стандартное отклонение')
    plt.title('Стандартное отклонение J от уровня шума для разного числа замеров')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/std_deviation.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_dir}/std_deviatio.pdf", bbox_inches='tight')
    plt.close()


def plot_boxplot(df, output_dir, hue, title):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="sigma", y="deviation_metric", data=df, hue=hue,
        palette="muted", gap = 0.1)
    plt.title(title, fontsize=18)
    plt.xlabel("Уровень шума σ", fontsize=16)
    plt.ylabel("J", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=False, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplot_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/boxplot_graph.pdf", bbox_inches='tight')
    plt.close()


def plot_violinplot(df, output_dir, hue, title):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="sigma", y="deviation_metric", hue=hue,
        data=df, palette="muted", density_norm='width', inner="box"
    )
    plt.title(title, fontsize=18)
    plt.xlabel("Уровень шума σ", fontsize=16)
    plt.ylabel("J", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(frameon=False, fontsize=13)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/violinplot_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/violinplot_graph.pdf", bbox_inches='tight')
    plt.close()


def plot_barplot(df, output_dir, title):
    set_plot_style()

    df_grouped = df.groupby("Метод оптимизации").agg(
        mean_J=("deviation_metric", "mean"),
        mean_E=("E", "mean"),
        std_metric=("deviation_metric", "std"),
        mean_time=("elapsed_time", "mean"),
        std_time=("elapsed_time", "std")
    ).reset_index()

    df_J_sorted = df_grouped.sort_values("mean_J")
    df_time_sorted = df_grouped.sort_values("mean_time")
    df_E_sorted = df_grouped.sort_values("mean_E")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    bars1 = ax1.bar(
        df_J_sorted["Метод оптимизации"],
        df_J_sorted["mean_J"],
        color=sns.color_palette("crest", len(df_J_sorted))
    )
    ax1.set_title("Среднее J")
    ax1.set_ylabel("J (%)")
    ax1.tick_params(axis='x', rotation=30)

    y_max_1 = max([bar.get_height() for bar in bars1])
    ax1.set_ylim(0, y_max_1 * 1.15)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + y_max_1 * 0.02,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    bars2 = ax2.bar(
        df_time_sorted["Метод оптимизации"],
        df_time_sorted["mean_time"],
        color=sns.color_palette("flare", len(df_time_sorted))
    )
    ax2.set_title("Среднее время выполнения")
    ax2.set_ylabel("Время (сек)")
    ax2.tick_params(axis='x', rotation=30)

    y_max_2 = max([bar.get_height() for bar in bars2])
    ax2.set_ylim(0, y_max_2 * 1.15)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height + y_max_2 * 0.02,
                 f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{output_dir}/barplot_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/barplot_graph.pdf", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        df_J_sorted["Метод оптимизации"],
        df_J_sorted["mean_J"],
        color=sns.color_palette("crest", len(df_J_sorted))
    )
    ax.set_title("Среднее J")
    ax.set_ylabel("J (%)")
    ax.tick_params(axis='x', rotation=30)

    y_max = max([bar.get_height() for bar in bars])
    ax.set_ylim(0, y_max * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + y_max * 0.02,
                f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{output_dir}/barplot_metric_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/barplot_metric_graph.pdf", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        df_E_sorted["Метод оптимизации"],
        df_E_sorted["mean_E"],
        color=sns.color_palette("crest", len(df_E_sorted))
    )
    ax.set_title("Среднее E")
    ax.set_ylabel("E")
    ax.tick_params(axis='x', rotation=30)

    y_max = max([bar.get_height() for bar in bars])
    ax.set_ylim(0, y_max * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + y_max * 0.02,
                f"{height:.3f}", ha='center', va='bottom', fontsize=10)

    plt.suptitle(title, fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{output_dir}/barplot_E_graph.png", dpi=300, format="png", bbox_inches='tight')
    plt.savefig(f"{output_dir}/barplot_E_graph.pdf", bbox_inches='tight')
    plt.close()


def plot_applicability_heatmap(df, output_path):
    set_plot_style()

    df["rounded_dev"] = df["mean_deviation"].round(2)

    x_vals = np.sort(df["Pe0"].unique())
    y_vals = np.sort(df["A"].unique())

    z_grid = np.full((len(y_vals), len(x_vals)), np.nan)

    for i, a in enumerate(y_vals):
        for j, pe in enumerate(x_vals):
            match = df[(df["Pe0"] == pe) & (df["A"] == a)]
            if not match.empty:
                z_grid[i, j] = match["rounded_dev"].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    c = ax.imshow(
        z_grid,
        cmap='Blues',
        interpolation='none',
        aspect='auto',
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        origin='lower'
    )

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("J", fontsize=14)

    ax.set_title("Карта применимости модели")
    ax.set_xlabel("Pe")
    ax.set_ylabel("A",)

    ax.set_xticks(np.linspace(x_vals.min(), x_vals.max(), min(10, len(x_vals))))
    ax.set_yticks(np.linspace(y_vals.min(), y_vals.max(), min(10, len(y_vals))))

    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300)
    plt.savefig(output_path, dpi=300)
    plt.close()