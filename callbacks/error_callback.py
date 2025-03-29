import dash
from dash import html
from dash.dependencies import Input, Output
from block.cache import get_boundaries_cached, perform_optimization_cached, extract_boundaries_cached
from optimizator.intervals import calculate_error_percentage
from optimizator.optimizer import compute_relative_error

def register_error_callback(app):
    @app.callback(
        Output('error-text', 'children'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_error(b_values, boundary_values, A, TG0, atg, sigma, N):
        found_left, found_right = extract_boundaries_cached(boundary_values, force_update=False)

        result, param_history, df_history, x_data, y_data = perform_optimization_cached(found_left, found_right, b_values, TG0, atg, A, sigma, N, force_update=False)

        regression_error = compute_relative_error(param_history, b_values)

        optimizator_error = calculate_error_percentage(
            boundary_values['left'], boundary_values['right'], found_left, found_right)

        return html.Div([
            html.H4("Результаты расчета погрешностей:"),
            html.P(f"Погрешность регрессии: {regression_error:.2f}%"),
            html.P(f"Погрешность оптимизации: {optimizator_error:.2f}%"),
            html.Hr(),
        ])

        # except Exception as e:
        #     return html.Div([
        #         html.P("Ошибка при расчете погрешностей:", style={'color': 'black'}),
        #         html.P(str(e))
        #     ])

