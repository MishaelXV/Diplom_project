import os
import pandas as pd
import numpy as np
import warnings
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from main_block.data import generate_data, noize_data
from main_block.main_functions import main_func
from optimizator.bayes_optimizer import run_bayes_optimization
from optimizator.optimizer import run_optimization, calculate_deviation_metric
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms
from stability_tests.config import COMMON_CONSTANTS, STABILITY_CONFIGS
from stability_tests.plots import plot_boxplot, plot_violinplot, plot_mean_differences, plot_histograms, \
    plot_std_deviation, plot_barplot, plot_applicability_heatmap

warnings.filterwarnings('ignore', category=RuntimeWarning)

def save_dataframe(df, config_name):
    output_dir = STABILITY_CONFIGS[config_name]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{config_name}_results.csv"), index=False)

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

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, n, sigma, A,  model_ws, model_ms)

    if len(found_left) == 1:
        Pe_opt = [0]

    elif len(found_left) <= 2:
        Pe_opt = [Pe[0], 0]

    else:
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    deviation_metric = calculate_deviation_metric(x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)

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
    save_dataframe(df, "stability_std_N_samples")

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

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, n, sigma, A, model_ws, model_ms)
    start_time = time.time()

    if method == "bayes":
        Pe_opt = run_bayes_optimization(x_data, y_data_noize, found_left, found_right, Pe, TG0, atg, A, n_trials=100)

    else:
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                  boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, method)

    deviation_metric = calculate_deviation_metric(
        x_data, found_left, found_right,
        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)

    E = np.sum((main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right) - y_data_noize)**2)

    elapsed_time = time.time() - start_time

    return {
        "sigma": sigma,
        "N_samples": n,
        "deviation_metric": deviation_metric,
        "E": E,
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
    save_dataframe(df, "stability_optimaizers")

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

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, n, sigma, A, model_ws, model_ms)

    if len(found_left) == 1:
        Pe_opt = [0]

    elif len(found_left) <= 2:
        Pe_opt = [Pe[0], 0]

    else:
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                                         boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, method='nelder')

    deviation_metric = calculate_deviation_metric(x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)

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
    save_dataframe(df, "stability_N_samples")

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


def process_A_run(A_value, sigma, n):
    x_data, y_data = generate_data(
        boundary_dict["left"],
        boundary_dict["right"],
        Pe, TG0, atg, A_value, n)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, n, sigma, A_value, model_ws, model_ms)

    if len(found_left) == 1:
        Pe_opt = [0]

    elif len(found_left) <= 2:
        Pe_opt = [Pe[0], 0]

    else:
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                              boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A_value)

    deviation_metric = calculate_deviation_metric(x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)

    return {
        "sigma": sigma,
        "A": A_value,
        "deviation_metric": deviation_metric
    }


def run_A_analysis(variables, output_dir, n_jobs=5):
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
    save_dataframe(df, "stability_A")

    if "boxplot" in config["plots"]:
        plot_boxplot(df, output_dir, hue="A",
                     title="Распределение невязки решения для разных значений A")
    if "violinplot" in config["plots"]:
        plot_violinplot(df, output_dir, hue="A",
                        title="Распределение невязки решения для разных значений A")

    return df


def process_applicability_run(Pe_1, A_value, sigma, n, N_rnd):
    deviations = []

    Pe_list = [Pe_1]
    for _ in range(1, len(boundary_dict["left"])):
        Pe_list.append(0.5 * Pe_list[-1])

    for _ in range(N_rnd):
        x_data, y_data = generate_data(
            boundary_dict["left"],
            boundary_dict["right"],
            Pe_list, TG0, atg, A_value, n)
        y_data_noize = noize_data(y_data, sigma)

        found_left, found_right = get_boundaries(x_data, y_data_noize, Pe_list, n,
                                                 sigma, A_value, model_ws, model_ms)
        if len(found_left) == 1:
            Pe_opt = [0]

        elif len(found_left) <= 2:
            Pe_opt = [Pe[0], 0]

        else:
            Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                                  boundary_dict['left'], boundary_dict['right'],
                                                  Pe_list, TG0, atg, A_value)

        deviation_metric = calculate_deviation_metric(x_data, found_left, found_right,
                                        boundary_dict['left'], boundary_dict['right'], Pe, Pe_opt)

        deviations.append(deviation_metric)

    return {
        "Pe0": Pe_1,
        "A": A_value,
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
    save_dataframe(df, "stability_applicability_map")

    if STABILITY_CONFIGS["stability_applicability_map"]["plots"]:
        plot_applicability_heatmap(df, f"{output_dir}/applicability_map_heatmap.png")

    return df