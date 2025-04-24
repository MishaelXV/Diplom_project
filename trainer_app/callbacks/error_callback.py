from dash import html
from dash.dependencies import Input, Output

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

        return html.Div([
            html.H4("Результаты расчета погрешностей:"),
            html.P(f"Заданные границы: {boundaries_data['left_true'], boundaries_data['right_true']}"),
            html.P(f"Найденные границы: {boundaries_data['left'], boundaries_data['right']}"),
            html.P("Погрешность регрессии: NaN"),
            html.P("Погрешность оптимизации: NaN"),
            html.P("RMSE: NaN"),
            html.Hr(),
        ])

        # except Exception as e:
        #     return html.Div([
        #         html.P("Ошибка при расчете погрешностей:", style={'color': 'black'}),
        #         html.P(str(e))
        #     ])

