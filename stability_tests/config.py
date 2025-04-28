import numpy as np

STABILITY_CONFIGS = {
    "stability_std_N_samples": {
        "output_dir": "data/charts/stability_std_N_samples",
        "variables": {
            "sigma_values": np.linspace(0.001, 0.05, 100),
            "N_samples": [200, 300, 500],
            "N_rnd": 500
        },
        "plots": ["std_deviation", "mean_difference", "histograms"]
    },
    "stability_optimaizers": {
        "output_dir": "data/charts/stability_optimaizers",
        "variables": {
            "sigma_values": [0.001],
            "methods": ['slsqp', 'cobyla', 'leastsq', 'least_squares', 'bfgs', 'nelder', 'lbfgsb', 'powell'],
            "N_samples": 500,
            "N_rnd": 1000
        },
        "plots": ["barplot"]
    },
    "stability_N_samples": {
        "output_dir": "data/charts/stability_N_samples",
        "variables": {
            "sigma_values": [0.001, 0.005, 0.01],
            "N_samples": [200, 300, 500],
            "N_rnd": 1000
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_A": {
        "output_dir": "data/charts/stability_A",
        "variables": {
            "sigma_values": [0.001, 0.005, 0.01],
            "A_values": [10, 5, 1],
            "N_samples": 500,
            "N_rnd": 100
        },
        "plots": ["boxplot", "violinplot"]
    },
    "stability_applicability_map": {
        "output_dir": "data/charts/stability_applicability_map",
        "variables": {
            "Pe0_values": np.linspace(200, 20000, 100),
            "A_values": np.linspace(1, 10, 10),
            "sigma": 0.001,
            "N_samples": 300,
            "N_rnd": 500
        },
        "plots": True
    }
}

COMMON_CONSTANTS = {
    "zInf": 100000,
    "TG0": 1,
    "atg": 0.0001,
    "A": 5,
    "Pe": [5000, 2000, 1000, 0],
    "boundaries": {"left": [0, 150, 300, 450], "right": [100, 250, 400, 550]}
}

ENABLED_TESTS = [
"stability_std_N_samples",
"stability_applicability_map",
]
