import os
from matplotlib.animation import PillowWriter
from main_block.data import generate_data, noize_data
from animations.fit_ani import run_ani
from animations.fit_plot import plot_comparison
from animations.optimization_path_ani import plot_optimization_path
from animations.params_ani import plot_animated_Pe
from animations.residual_ani import plot_animated_residuals
from optimizator.optimizer import run_optimization
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms
from animations.constants import COMMON_CONSTANTS

boundary_dict = COMMON_CONSTANTS['boundaries']
Pe = COMMON_CONSTANTS['Pe']
N = COMMON_CONSTANTS['N']
sigma = COMMON_CONSTANTS['sigma']
TG0 = COMMON_CONSTANTS['TG0']
atg = COMMON_CONSTANTS['atg']
A = COMMON_CONSTANTS['A']

def main():
    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data_noize, Pe, N, sigma, A, model_ws, model_ms)

    Pe_opt, df_history = run_optimization(x_data, y_data, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    plot_comparison(x_data, y_data_noize, Pe_opt, found_left, found_right)
    fit_ani = run_ani(x_data, y_data_noize, df_history, found_left, found_right)
    residual_ani = plot_animated_residuals(df_history)
    Pe_ani = plot_animated_Pe(df_history)
    optimization_path_ani = plot_optimization_path(df_history, found_left, found_right, x_data)

    os.makedirs('charts', exist_ok=True)
    fit_ani.save('charts/fit_animation.gif', writer=PillowWriter(fps=5))
    residual_ani.save('charts/residuals_animation.gif', writer=PillowWriter(fps=5))
    Pe_ani.save('charts/pe_animation.gif', writer=PillowWriter(fps=5))
    optimization_path_ani.save('charts/optimization_path.gif', writer=PillowWriter(fps=5))


if __name__ == "__main__":
    main()
