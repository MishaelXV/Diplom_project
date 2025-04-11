import numpy as np
import pandas as pd
from lmfit import minimize, Parameters
from calculates_block.main_functions import main_func
from config import STABILITY_CONFIGS
from stability_tests.config import COMMON_CONSTANTS
from stability_tests.plots import plot_boxplot, plot_violinplot, plot_mean_differences, plot_histograms, \
    plot_std_deviation

def residuals(params, x, y):
    return main_func(params, x, **COMMON_CONSTANTS) - y


def initialize_params():
    params = Parameters()
    for i in range(len(COMMON_CONSTANTS["Pe"]) - 1):
        params.add(f'Pe_{i + 1}', value=1, min=0, max=5000)
    return params


def run_std_n_samples_analysis(params, variables, output_dir):
    results = []
    all_differences = [[] for _ in variables["N_samples"]]
    mean_differences = [[] for _ in variables["N_samples"]]

    config = STABILITY_CONFIGS["stability_std_N_samples"]

    for n_idx, n in enumerate(variables["N_samples"]):
        disp_results = []
        for sigma in variables["sigma_values"]:
            delta = []
            for _ in range(variables["N_rnd"]):
                x_data = np.linspace(COMMON_CONSTANTS["b"][0], COMMON_CONSTANTS["c"][-1], n)
                y_data = main_func(
                    {f'Pe_{i+1}': COMMON_CONSTANTS["Pe"][i] for i in range(len(COMMON_CONSTANTS["Pe"])-1)},
                    x_data, **COMMON_CONSTANTS
                ) + np.random.normal(0, sigma, x_data.size)

                result = minimize(residuals, params, args=(x_data, y_data), method='leastsq')

                for i, param in enumerate(result.params.values()):
                    delta_val = abs(param.value - COMMON_CONSTANTS["Pe"][i])
                    delta.append(delta_val)
                    all_differences[n_idx].append(delta_val)

            disp_results.append(np.std(delta))
            mean_differences[n_idx].append(np.mean(delta))
        results.append(disp_results)

    # Генерация графиков
    if "std_deviation" in config["plots"]:
        plot_std_deviation(results, variables, output_dir)
    if "mean_difference" in config["plots"]:
        plot_mean_differences(mean_differences, variables, output_dir)
    if "histograms" in config["plots"]:
        plot_histograms(all_differences, variables["N_samples"], output_dir)

def run_optimizers_analysis(params, variables, output_dir):
    results = []
    all_data = []
    all_differences = [[] for _ in variables["methods"]]
    mean_differences = [[] for _ in variables["methods"]]

    config = STABILITY_CONFIGS["stability_optimaizers"]

    for n_idx, n in enumerate(variables["methods"]):
        disp_results = []
        for sigma in variables["sigma_values"]:
            sigma_differences = []
            for _ in range(variables["N_rnd"]):
                x_data = np.linspace(COMMON_CONSTANTS["b"][0], COMMON_CONSTANTS["c"][-1], variables["N_samples"])
                y_data = main_func(
                    {f'Pe_{i + 1}': COMMON_CONSTANTS["Pe"][i] for i in range(len(COMMON_CONSTANTS["Pe"]) - 1)},
                    x_data, **COMMON_CONSTANTS
                ) + np.random.normal(0, sigma, x_data.size)

                result = minimize(residuals, params, args=(x_data, y_data), method=n)

                for i, param in enumerate(result.params.values()):
                    delta = abs(param.value - COMMON_CONSTANTS["Pe"][i])
                    all_data.append({"sigma": sigma, "delta": delta, "Метод оптимизации": n})
                    sigma_differences.append(delta)

            all_differences[n_idx].extend(sigma_differences)
            mean_differences[n_idx].append(np.mean(sigma_differences))
        disp_results.append(np.std(sigma_differences))
        results.append(disp_results)

    df = pd.DataFrame(all_data)
    df['Метод оптимизации'] = df['Метод оптимизации'].replace({
        'leastsq': 'lm'})

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="Метод оптимизации",
                     title="Распределение отклонений профиля для разных методов оптимизации")

    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="Метод оптимизации",
                        title="Распределение отклонений профиля для разных методов оптимизации")


def run_n_samples_analysis(params, variables, output_dir):
    results = []
    all_data = []
    all_differences = [[] for _ in variables["N_samples"]]
    mean_differences = [[] for _ in variables["N_samples"]]

    config = STABILITY_CONFIGS["stability_N_samples"]

    for n_idx, n in enumerate(variables["N_samples"]):
        disp_results = []
        for sigma in variables["sigma_values"]:
            sigma_differences = []
            for _ in range(variables["N_rnd"]):
                x_data = np.linspace(COMMON_CONSTANTS["b"][0], COMMON_CONSTANTS["c"][-1], n)
                y_data = main_func(
                    {f'Pe_{i + 1}': COMMON_CONSTANTS["Pe"][i] for i in range(len(COMMON_CONSTANTS["Pe"]) - 1)},
                    x_data, **COMMON_CONSTANTS
                ) + np.random.normal(0, sigma, x_data.size)

                result = minimize(residuals, params, args=(x_data, y_data), method='leastsq')

                for i, param in enumerate(result.params.values()):
                    delta = abs(param.value - COMMON_CONSTANTS["Pe"][i])
                    all_data.append({"sigma": sigma, "delta": delta, "число замеров": n})
                    sigma_differences.append(delta)

            all_differences[n_idx].extend(sigma_differences)
            mean_differences[n_idx].append(np.mean(sigma_differences))
        disp_results.append(np.std(sigma_differences))
        results.append(disp_results)

    df = pd.DataFrame(all_data)

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="число замеров",
                     title="Распределение отклонений профиля для разного числа замеров")

    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="число замеров",
                        title="Распределение отклонений профиля для разного числа замеров")

    if "mean_differences" in config["plots"]:
        plot_mean_differences(mean_differences, variables, output_dir)

    if "histograms" in config["plots"]:
        plot_histograms(all_differences, variables["N_samples"], output_dir)


def run_A_analysis(params, variables, output_dir):
    results = []
    mean_differences = []
    all_data = []

    config = STABILITY_CONFIGS["stability_A"]

    for a_value in variables["A_values"]:
        disp_results = []
        mean_results = []
        for sigma in variables["sigma_values"]:
            deviations = []
            for _ in range(variables["N_rnd"]):

                original_A = COMMON_CONSTANTS["A"]
                COMMON_CONSTANTS["A"] = a_value

                x_data = np.linspace(COMMON_CONSTANTS["b"][0], COMMON_CONSTANTS["c"][-1], variables["N_samples"])
                y_data = main_func(
                    {f'Pe_{i + 1}': COMMON_CONSTANTS["Pe"][i] for i in range(len(COMMON_CONSTANTS["Pe"]) - 1)},
                    x_data, **COMMON_CONSTANTS
                ) + np.random.normal(0, sigma, x_data.size)

                result = minimize(residuals, params, args=(x_data, y_data), method='leastsq')

                for i, param in enumerate(result.params.values()):
                    delta = abs(param.value - COMMON_CONSTANTS["Pe"][i])
                    all_data.append({"sigma": sigma, "delta": delta, "A": a_value})
                    deviations.append(delta)

                COMMON_CONSTANTS["A"] = original_A

        disp_results.append(np.std(deviations))
        mean_results.append(np.mean(deviations))
        results.append(disp_results)
        mean_differences.append(mean_results)

    df = pd.DataFrame(all_data)
    df['sigma'] = df['sigma'].round(4)

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="A", title="Распределение отклонений профиля для разных значений A")

    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="A", title="Распределение отклонений профиля для разных значений A")