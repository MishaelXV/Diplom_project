import pandas as pd
import numpy as np
import warnings
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from lmfit import minimize
from joblib import Parallel, delayed
from calculates_block.data import generate_data, noize_data
from optimizator.optimizer import run_optimization, calculate_deviation_metric, optimization_residuals, \
    optimization_callback, create_parameters
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms
from stability_tests.config import COMMON_CONSTANTS
from config import STABILITY_CONFIGS
from stability_tests.plots import plot_boxplot, plot_violinplot, plot_mean_differences, plot_histograms, \
    plot_std_deviation, plot_barplot, plot_applicability_heatmap

warnings.filterwarnings('ignore', category=RuntimeWarning)
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

# Общие константы
zInf = COMMON_CONSTANTS['zInf']
TG0 = COMMON_CONSTANTS['TG0']
atg = COMMON_CONSTANTS['atg']
A = COMMON_CONSTANTS['A']
Pe = COMMON_CONSTANTS['Pe']
boundary_dict = COMMON_CONSTANTS['boundaries']

def process_std_run(n, sigma):
    x_data, y_data = generate_data(
        boundary_dict["left"],
        boundary_dict["right"],
        Pe, TG0, atg, A, n)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, n, sigma, A,  model_ws, model_ms)

    result, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    deviation_metric = calculate_deviation_metric(result.params, x_data, found_left, found_right,
                                                  boundary_dict['left'], boundary_dict['right'], Pe)

    return {
        "N_samples": n,
        "sigma": sigma,
        "deviation_metric": deviation_metric
    }


def run_std_n_samples_analysis(variables, output_dir, n_jobs=-1):
    config = STABILITY_CONFIGS["stability_std_N_samples"]

    tasks = [(n, sigma)
             for n in variables["N_samples"]
             for sigma in variables["sigma_values"]
             for _ in range(variables["N_rnd"])]

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(process_std_run)(*task)
            for task in tqdm(tasks, desc="Standard N Samples Analysis")
        )

    df = pd.DataFrame(results)

    if "std_deviation" in config["plots"]:
        plot_std_deviation(df, variables, output_dir)
    if "mean_difference" in config["plots"]:
        plot_mean_differences(df, variables, output_dir)
    if "histograms" in config["plots"]:
        plot_histograms(df, variables["N_samples"], output_dir)

    return df


def process_optimizer_run(method, sigma, n):
    x_data, y_data = generate_data(
        boundary_dict["left"],
        boundary_dict["right"],
        Pe, TG0, atg, A, n)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, n, sigma, A, model_ws, model_ms)
    params = create_parameters(found_left, COMMON_CONSTANTS['Pe'][0])

    param_history = []

    start_time = time.time()

    result = minimize(
        lambda p, x, y: optimization_residuals(p, x, y, TG0, atg, A, Pe, found_left, found_right),
        params,
        args=(x_data, y_data_noize),
        method=method,
        iter_cb=lambda p, i, r, *a, **kw: optimization_callback(
            p, i, r, param_history, x_data, found_left, found_right, boundary_dict["left"], boundary_dict["right"], Pe),
        nan_policy='omit'
    )

    elapsed_time = time.time() - start_time

    deviation_metric = calculate_deviation_metric(result.params, x_data, found_left, found_right, boundary_dict['left'], boundary_dict['right'], Pe)

    return {
        "sigma": sigma,
        "N_samples": n,
        "deviation_metric": deviation_metric,
        "elapsed_time": elapsed_time,
        "Метод оптимизации": method
    }


def run_optimizers_analysis(variables, output_dir, n_jobs=-1):
    config = STABILITY_CONFIGS["stability_optimaizers"]

    tasks = [(method, sigma, variables["N_samples"])
             for method in variables["methods"]
             for sigma in variables["sigma_values"]
             for _ in range(variables["N_rnd"])]

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(process_optimizer_run)(*task)
            for task in tqdm(tasks, desc="Optimizers Analysis")
        )

    df = pd.DataFrame(results)
    df['Метод оптимизации'] = df['Метод оптимизации'].replace({'leastsq': 'lm'})

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="Метод оптимизации",
                     title="Распределение невязки решения для разных методов оптимизации")
    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="Метод оптимизации",
                        title="Распределение невязки решения для разных методов оптимизации")
    if "barplot" in config["plots"]:
        plot_barplot(df, output_dir, title="Сравнение различных методов оптимизации")

    return df


def process_n_samples_run(n, sigma):
    x_data, y_data = generate_data(
        boundary_dict["left"],
        boundary_dict["right"],
        Pe, TG0, atg, A, n)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, n, sigma, A, model_ws, model_ms)

    result, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                                         boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    deviation_metric = calculate_deviation_metric(result.params, x_data, found_left, found_right,
                                                  boundary_dict['left'], boundary_dict['right'], Pe)

    return {
        "N_samples": n,
        "sigma": sigma,
        "deviation_metric": deviation_metric
    }


def run_n_samples_analysis(variables, output_dir, n_jobs=-1):
    config = STABILITY_CONFIGS["stability_N_samples"]

    tasks = [(n, sigma)
             for n in variables["N_samples"]
             for sigma in variables["sigma_values"]
             for _ in range(variables["N_rnd"])]

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(process_n_samples_run)(*task)
            for task in tqdm(tasks, desc="N Samples Analysis")
        )

    df = pd.DataFrame(results)

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="N_samples",
                     title="Распределение невязки решения для разного числа замеров")
    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="N_samples",
                        title="Распределение невязки решения для разного числа замеров")
    if "mean_differences" in config["plots"]:
        plot_mean_differences(df, variables, output_dir)
    if "histograms" in config["plots"]:
        plot_histograms(df, variables["N_samples"], output_dir)

    return df


def process_A_run(a_value, sigma, n):
    original_A = A
    COMMON_CONSTANTS["A"] = a_value

    x_data, y_data = generate_data(
        boundary_dict["left"],
        boundary_dict["right"],
        Pe, TG0, atg, COMMON_CONSTANTS["A"], n)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, n, sigma, COMMON_CONSTANTS["A"], model_ws,
                                             model_ms)

    result, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    deviation_metric = calculate_deviation_metric(result.params, x_data, found_left, found_right,
                                                  boundary_dict['left'], boundary_dict['right'], Pe)

    COMMON_CONSTANTS["A"] = original_A

    return {
        "sigma": sigma,
        "A": a_value,
        "deviation_metric": deviation_metric
    }


def run_A_analysis(variables, output_dir, n_jobs=-1):
    config = STABILITY_CONFIGS["stability_A"]

    tasks = [(a_value, sigma, variables["N_samples"])
             for a_value in variables["A_values"]
             for sigma in variables["sigma_values"]
             for _ in range(variables["N_rnd"])]

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(process_A_run)(*task)
            for task in tqdm(tasks, desc="A Parameter Analysis")
        )

    df = pd.DataFrame(results)
    df['sigma'] = df['sigma'].round(4)

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="A",
                     title="Распределение невязки решения для разных значений A")
    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="A",
                        title="Распределение невязки решения для разных значений A")

    return df


def process_applicability_run(pe0_value, a_value, sigma, n, N_rnd):
    deviations = []
    original_A = COMMON_CONSTANTS["A"]
    Pe_mod = Pe.copy()
    Pe_mod[0] = pe0_value
    COMMON_CONSTANTS["A"] = a_value

    for _ in range(N_rnd):
        x_data, y_data = generate_data(
            boundary_dict["left"],
            boundary_dict["right"],
            Pe, TG0, atg, A, n)
        y_data_noize = noize_data(y_data, sigma)

        found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, n, sigma, A, model_ws, model_ms)

        result, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                              boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

        deviation_metric = calculate_deviation_metric(result.params, x_data, found_left, found_right,
                                                      boundary_dict['left'], boundary_dict['right'], Pe)

        deviations.append(deviation_metric)

    COMMON_CONSTANTS["A"] = original_A

    return {
        "Pe0": pe0_value,
        "A": a_value,
        "mean_deviation": np.mean(deviations)
    }


def run_applicability_map_analysis(variables, output_dir, n_jobs=-1):
    pe0_values = variables["Pe0_values"]
    a_values = variables["A_values"]
    sigma = variables["sigma"]
    n = variables["N_samples"]
    N_rnd = variables["N_rnd"]

    tasks = [
        (pe0, a_val, sigma, n, N_rnd)
        for pe0 in pe0_values
        for a_val in a_values
    ]

    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(process_applicability_run)(*task)
            for task in tqdm(tasks, desc="Applicability Map Analysis")
        )

    df = pd.DataFrame(results)

    df.to_csv(f"{output_dir}/applicability_map_results.csv", index=False)

    if STABILITY_CONFIGS["stability_applicability_map"]["plots"]:
        plot_applicability_heatmap(df, f"{output_dir}/applicability_map_heatmap.png")

    return df