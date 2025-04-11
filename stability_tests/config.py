import numpy as np

STABILITY_CONFIGS = {
    "stability_std_N_samples": {
        "output_dir": "data/charts/stability_std_N_samples",
        "variables": {
            "sigma_values": np.linspace(0.01, 0.1, 10),
            "N_samples": [20, 50, 100],
            "N_rnd": 1
        },
        "plots": ["std_deviation", "mean_difference", "histograms"]
    },
    "stability_optimaizers": {
        "output_dir": "data/charts/stability_optimaizers",
        "variables": {
            "sigma_values": [0.01, 0.05, 0.1],
            "methods": ['leastsq', 'nelder', 'powell'],
            "N_samples": 50,
            "N_rnd": 1
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_N_samples": {
        "output_dir": "data/charts/stability_N_samples",
        "variables": {
            "sigma_values": [0.01, 0.05, 0.1],
            "N_samples": [100, 50, 20],
            "N_rnd": 1
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_A": {
        "output_dir": "data/charts/stability_A",
        "variables": {
            "sigma_values": [0.01, 0.05, 0.1],
            "A_values": [5, 3, 1],
            "N_samples": 150,
            "N_rnd": 1
        },
        "plots": ["boxplot", "violinplot"]
    }
}


# Общие константы
COMMON_CONSTANTS = {
    "zInf": 100000,
    "TG0": 1,
    "atg": 0.0001,
    "A": 5,
    "b": [0, 150, 300],
    "c": [100, 250, 400],
    "Pe": [2000, 1000, 0]
}