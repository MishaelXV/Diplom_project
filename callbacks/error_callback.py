from dash import html
from dash.dependencies import Input, Output
from regression.intervals import calculate_error_percentage
from optimizator.optimizer import compute_relative_error
import numpy as np

def register_error_callback(app):
    @app.callback(
        Output('error-text', 'children'),
        Input('optimization-cache', 'data'),
         Input('boundaries-cache', 'data'),
        prevent_initial_call=True
    )
    def update_error(optimization_data, boundaries_data):
        found_left = boundaries_data['left']
        found_right = boundaries_data['right']

        param_history = [
            (item['params'], item['residual'])
            for item in optimization_data['param_history']
        ]

        optimizator_error = compute_relative_error(param_history, optimization_data['true_Pe'])
        regression_error = calculate_error_percentage(
            boundaries_data['left_true'], boundaries_data['right_true'], found_left, found_right)

        return html.Div([
            html.H4("Результаты расчета погрешностей:"),
            html.P(f"Погрешность регрессии: {regression_error:.2f}%"),
            html.P(f"Погрешность оптимизации: {optimizator_error:.2f}%"),
            html.P(f"RMSE: {np.sqrt(regression_error*regression_error + optimizator_error*optimizator_error):.2f}%"),
            html.Hr(),
        ])

        # except Exception as e:
        #     return html.Div([
        #         html.P("Ошибка при расчете погрешностей:", style={'color': 'black'}),
        #         html.P(str(e))
        #     ])

