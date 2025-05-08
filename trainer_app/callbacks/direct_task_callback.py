import dash
import plotly.graph_objects as go
from dash.dependencies import Input, Output, ALL
from main_block.data import generate_data, noize_data
from main_block.data import save_temperature_values
from trainer_app.components.graphs import create_figure_direct_task
from trainer_app.components.valid_inputs_of_params import validate_inputs

def register_direct_task_callback(app):
    @app.callback(
        [Output('quadratic-graph', 'figure'),
         Output('plot-area-container', 'style')],
        [Input('a-input', 'value'),
         Input({'type': 'b-input', 'index': ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value'),
         Input({'type': 'submit-boundary', 'index': ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def update_graph_and_visibility(a, b_values, boundary_values, A, TG0, atg, sigma, N, clicks):
        boundary_clicked = any(clicks) if clicks else False

        params = [a, A, TG0, atg, sigma, N]
        params_valid = all(param is not None and (isinstance(param, (int, float)) and param >= 0) for param in params)

        b_values_valid = all(
            b_values is not None and (isinstance(b, (int, float)) and b >= 0 for b in b_values) if b_values else False)

        if not (params_valid and b_values_valid):
            return dash.no_update, {'display': 'none'}

        try:
            b_values, left_boundary, right_boundary = validate_inputs(a, b_values, boundary_values, A, TG0, atg, sigma)
            z_all, T_all = generate_data(left_boundary, right_boundary, b_values, TG0, atg, A, N)
            T_all_noisy = noize_data(T_all, sigma)

            save_temperature_values(T_all, 'temperatura_values.txt')

            fig = create_figure_direct_task(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a)

            container_style = {
                'display': 'flex' if boundary_clicked else 'none',
                'width': '65%',
                'height': '725px',
                'marginLeft': '15px',
                'marginRight': '15px',
                'marginTop': '15px',
                'backgroundColor': '#111111',
                'border': '1px solid #444444',
                'padding': '0',
                'flexDirection': 'column',
                'alignItems': 'stretch',
            }

            return fig, container_style

        except (ValueError, TypeError, Exception) as e:
            return go.Figure(layout=go.Layout(title="Ошибка в данных")), {'display': 'none'}