import dash
from dash.dependencies import Input, Output
from calculates_block.calculates import perform_optimization
from components.boundaries import extract_boundaries
from regression.intervals import get_boundaries
from optimizator.optimizer import generate_data

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

        left, right = get_boundaries(boundary_data, b_values, N, sigma, TG0, atg, A)
        left_true, right_true = extract_boundaries(boundary_data)

        result, param_history, df_history, x_data, y_data = perform_optimization(
            left, right, b_values, TG0, atg, A, sigma, N
        )

        x_data_true, y_data_true = generate_data(left_true, right_true, b_values, 1000000, TG0, atg, A, sigma, N)

        boundaries_cache = {
            'left': left,
            'right': right,
            'left_true': left_true,
            'right_true': right_true,
            'x_data_true': x_data_true,
            'y_data_true': y_data_true,
        }

        serializable_param_history = [
            {
                'params': dict(params),
                'residual': float(residual)
            }
            for params, residual in param_history
        ]

        df_history_dict = {
            'columns': df_history.columns.tolist(),
            'data': df_history.values.tolist()
        }

        optimization_cache = {
            'params': {k: v.value for k, v in result.params.items()},
            'true_Pe': b_values,
            'success': bool(result.success),
            'message': str(result.message),
            'param_history': serializable_param_history,
            'df_history': df_history_dict,
            'x_data': x_data.tolist(),
            'y_data': y_data.tolist(),
            'chisqr': float(result.chisqr) if hasattr(result, 'chisqr') else None,
            'redchi': float(result.redchi) if hasattr(result, 'redchi') else None
        }

        return optimization_cache, boundaries_cache