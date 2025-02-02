import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from components.boundaries import extract_boundaries
from components.graphs import create_figure_animation, generate_frames
from block.calculates import perform_optimization

def register_animation_callback(app):
    @app.callback(
        Output('animation-container', 'style'),
        Output('animation-graph', 'figure'),
        Input('solve_inverse_task', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_animation(n_clicks, b_values, boundary_values, A, TG0, atg, sigma, N):
        if not n_clicks:
            return {'display': 'none'}, dash.no_update
        if len(b_values) == 1:
            return {'display': 'none'}, dash.no_update

        try:
            left_boundary, right_boundary = extract_boundaries(boundary_values)
            result, param_history, df_history, x_data, y_data = perform_optimization(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)

            frames = generate_frames(param_history, x_data, y_data, left_boundary, right_boundary, TG0, atg, A, b_values)
            fig = create_figure_animation(frames, x_data, param_history, left_boundary, right_boundary, TG0, atg, A, b_values, y_data)

            return {
                'display': 'block',
                'width': '98.5%',
                'height': '450px',
                'border': '1px solid #ccc',
                'padding': '10px',
                'boxSizing': 'border-box',
                'backgroundColor': '#ffffff'
            }, fig

        except Exception as e:
            return {'display': 'none'}, go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))