import os
from config import STABILITY_CONFIGS, ENABLED_TESTS
from stability_tests.analysis import (
    run_optimizers_analysis,
    run_n_samples_analysis,
    run_A_analysis,
    run_std_n_samples_analysis,
    run_applicability_map_analysis
)

def run_tests(config_name, config):
    print(f"\nRunning analysis: {config_name}")
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    variables = config["variables"]

    if config_name == "stability_std_N_samples":
        return run_std_n_samples_analysis(variables, output_dir)
    elif config_name == "stability_optimaizers":
        return run_optimizers_analysis(variables, output_dir)
    elif config_name == "stability_N_samples":
        return run_n_samples_analysis(variables, output_dir)
    elif config_name == "stability_A":
        return run_A_analysis(variables, output_dir)
    elif config_name == "stability_applicability_map":
        return run_applicability_map_analysis(variables, output_dir)
    else:
        raise ValueError(f"Unknown config: {config_name}")


if __name__ == "__main__":
    for config_name in ENABLED_TESTS:
        if config_name not in STABILITY_CONFIGS:
            print(f"Warning: '{config_name}' not found in STABILITY_CONFIGS, skipping.")
            continue
        run_tests(config_name, STABILITY_CONFIGS[config_name])
    print("All tests passed.")