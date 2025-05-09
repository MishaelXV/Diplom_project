import numpy as np
import itertools
from dash import Input, Output
from optimizator.optimizer import calculate_deviation_metric
from trainer_app.components.graphs import build_parallel_coordinates_figure
from trainer_app.components.support_functions import residuals

def register_parallel_graph_callback(app):
    @app.callback(
        Output('parallel-coordinates-graph', 'figure'),
        [Input('optimization-cache', 'data'),
         Input('boundaries-cache', 'data')],
        [Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value')],
        prevent_initial_call=True
    )
    def update_parallel_graph(opt_cache, boundaries_cache, A_val, TG0, atg):
        if opt_cache is None or boundaries_cache is None:
            return {}

        # Данные из кэша
        x_data = boundaries_cache['x_data_true']
        y_data = boundaries_cache['y_data_true']
        left_bounds = np.array(boundaries_cache['left'])
        right_bounds = np.array(boundaries_cache['right'])
        true_left = boundaries_cache['left_true']
        true_right = boundaries_cache['right_true']
        Pe_true = opt_cache['true_Pe']

        n = len(Pe_true)
        Pe_start = Pe_true[0]
        Pe_end = Pe_true[-1]
        num_opt_params = n - 2

        if num_opt_params <= 0:
            return {}

        A_grid = np.arange(1, 11).astype(float)

        pe_values = np.linspace(500, 2500, 10)

        inner_pe_grids = list(itertools.product(*[pe_values] * num_opt_params))

        valid_pe_combos = [
            combo for combo in inner_pe_grids
            if all(combo[i] <= combo[i - 1] for i in range(1, len(combo)))
        ]

        data = []
        for A in A_grid:
            for inner_pe in valid_pe_combos:
                Pe_opt = [Pe_start] + list(inner_pe) + [Pe_end]

                E = np.sum(residuals(x_data, y_data, TG0, atg, A, Pe_opt, left_bounds, right_bounds) ** 2)
                J = calculate_deviation_metric(x_data, left_bounds, right_bounds, true_left, true_right, Pe_true, Pe_opt)

                row = {'A': A, 'E': E, 'J': J}
                for i, pe in enumerate(Pe_opt[1:-1]):
                    row[f'Pe_{i + 2}'] = pe
                data.append(row)

        fig = build_parallel_coordinates_figure(data)
        return fig