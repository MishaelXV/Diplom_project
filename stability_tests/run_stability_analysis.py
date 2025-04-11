import os
from config import STABILITY_CONFIGS
from stability_tests.analysis import run_std_n_samples_analysis, run_optimizers_analysis, run_n_samples_analysis, \
    run_A_analysis, initialize_params

def run_analysis(config_name, config):
    print(f"\nRunning analysis: {config_name}")
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    params = initialize_params()
    variables = config["variables"]

    if config_name == "stability_std_N_samples":
        return run_std_n_samples_analysis(params, variables, output_dir)
    elif config_name == "stability_optimaizers":
        return run_optimizers_analysis(params, variables, output_dir)
    elif config_name == "stability_N_samples":
        return run_n_samples_analysis(params, variables, output_dir)
    elif config_name == "stability_A":
        return run_A_analysis(params, variables, output_dir)


if __name__ == "__main__":
    for config_name, config in STABILITY_CONFIGS.items():
        run_analysis(config_name, config)
    print("All tests passed!")