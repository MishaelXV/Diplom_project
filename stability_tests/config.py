import numpy as np

STABILITY_CONFIGS = {
    "stability_std_N_samples": {
        "output_dir": "data/charts/stability_std_N_samples",
        "variables": {
            "sigma_values": np.linspace(0.0001, 0.01, 50),
            "N_samples": [500, 800, 1000],
            "N_rnd": 100
        },
        "plots": ["std_deviation", "mean_difference", "histograms"]
    },
    "stability_optimaizers": {
        "output_dir": "data/charts/stability_optimaizers",
        "variables": {
            "sigma_values": [0.001],
            "methods": ['leastsq', 'nelder', 'powell'],
            "N_samples": 500,
            "N_rnd": 100
        },
        "plots": ["barplot"]
    },
    "stability_N_samples": {
        "output_dir": "data/charts/stability_N_samples",
        "variables": {
            "sigma_values": [0.0001, 0.001, 0.005],
            "N_samples": [500, 800, 1000],
            "N_rnd": 100
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_A": {
        "output_dir": "data/charts/stability_A",
        "variables": {
            "sigma_values": [0.0001, 0.001, 0.005],
            "A_values": [10, 5, 3],
            "N_samples": 500,
            "N_rnd": 100
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_applicability_map": {
        "output_dir": "data/charts/stability_applicability_map",
        "variables": {
            "Pe0_values": np.linspace(2000, 10000, 30),
            "A_values": np.linspace(1, 10, 30),
            "sigma": 0.001,
            "N_samples": 200,
            "N_rnd": 1
        },
        "plots": True
    }
}

COMMON_CONSTANTS = {
    "zInf": 100000,
    "TG0": 1,
    "atg": 0.0001,
    "A": 5,
    "Pe": [2000, 1000, 0],
    "boundaries": {"left": [0, 150, 300], "right": [100, 250, 400]}
}

ENABLED_TESTS = [
"stability_optimaizers",
"stability_std_N_samples",
"stability_N_samples",
"stability_A",
]
