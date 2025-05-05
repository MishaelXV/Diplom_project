import dash
from dash.dependencies import Input, Output
from optimizator.optimizer import run_optimization
from regression.global_models import model_ws, model_ms
from trainer_app.components.support_functions import extract_boundaries
from regression.find_intervals import get_boundaries
from main_block.data import generate_data, noize_data

def register_cache_callback(app):
    @app.callback(
        Output('optimization-cache', 'data'),
        Output('boundaries-cache', 'data'),
        Input('solve_inverse_task', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')],
        prevent_initial_call=True
    )
    def compute_and_cache(n_clicks, b_values, boundary_data, A, TG0, atg, sigma, N):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        left_true, right_true = extract_boundaries(boundary_data)
        x_data, y_data = generate_data(left_true, right_true, b_values, TG0, atg, A, N)
        y_data_noize = noize_data(y_data, sigma)

        found_left, found_right = get_boundaries(x_data, y_data_noize, b_values, N, sigma, A, model_ws, model_ms)
        Pe_opt, df_history = run_optimization(x_data, y_data_noize, found_left, found_right,
                                              boundary_data['left'], boundary_data['right'], b_values, TG0, atg, A)
        boundaries_cache = {
            'left': found_left,
            'right': found_right,
            'left_true': left_true,
            'right_true': right_true,
            'x_data_true': x_data,
            'y_data_true': y_data_noize,
        }

        df_history_dict = {
            'columns': df_history.columns.tolist(),
            'data': df_history.values.tolist()
        }

        optimization_cache = {
            'found_Pe': Pe_opt,
            'true_Pe': b_values,
            'df_history': df_history_dict,
            'x_data': x_data.tolist(),
            'y_data': y_data.tolist(),
            'fixed_pe': {
                'first': b_values[0],
                'last': b_values[-1]
            }
        }

        return optimization_cache, boundaries_cache