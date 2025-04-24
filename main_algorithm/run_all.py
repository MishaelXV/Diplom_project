from calculates_block.data import generate_data, noize_data
from main_algorithm.fit_ani import run_ani
from main_algorithm.fit_plot import plot_comparison
from main_algorithm.optimization_path_ani import plot_optimization_path
from main_algorithm.params_ani import plot_animated_Pe
from main_algorithm.residual_ani import plot_animated_residuals
from optimizator.optimizer import run_optimization
from regression.find_intervals import get_boundaries
from regression.global_models import model_ws, model_ms
from main_algorithm.constants import COMMON_CONSTANTS

def main():
    boundary_dict = COMMON_CONSTANTS['boundaries']
    Pe = COMMON_CONSTANTS['Pe']
    N = COMMON_CONSTANTS['N']
    sigma = COMMON_CONSTANTS['sigma']
    TG0 = COMMON_CONSTANTS['TG0']
    atg = COMMON_CONSTANTS['atg']
    A = COMMON_CONSTANTS['A']

    x_data, y_data = generate_data(boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A, N)
    y_data_noize = noize_data(y_data, sigma)

    found_left, found_right = get_boundaries(x_data, y_data, y_data_noize, Pe, N, sigma, A, model_ws, model_ms)

    result, df_history = run_optimization(x_data, y_data, found_left, found_right,
                                          boundary_dict['left'], boundary_dict['right'], Pe, TG0, atg, A)

    # fit_ani = run_ani(found_left, found_right, df_history, x_data, y_data_noize)
    ani_residuals = plot_animated_residuals(df_history)
    ani_pe = plot_animated_Pe(df_history)
    plot_final = plot_comparison(x_data, y_data_noize, result.params, found_left, found_right)
    # ani_optimization_path = plot_optimization_path(df_history, Pe)

    # os.makedirs('animations', exist_ok=True)
    # anim.save('animations/fit_animation.gif', writer=PillowWriter(fps=5))
    # ani_residuals.save('animations/residuals_animation.gif', writer=PillowWriter(fps=5))
    # ani_pe.save('animations/pe_animation.gif', writer=PillowWriter(fps=5))
    # ani_optimization_path.save('animations/optimization_path.gif', writer=PillowWriter(fps=5))


if __name__ == "__main__":
    main()
