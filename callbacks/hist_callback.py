import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from block.calculates import perform_optimization
from callbacks.boundaries import extract_boundaries
from callbacks.graphs import create_histogram

def register_hist_callback(app):
    @app.callback(
        Output('hist-graph', 'figure'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_hist(b_values, boundary_values, A, TG0, atg, sigma, N):
        try:
            left_boundary, right_boundary = extract_boundaries(boundary_values)
            result, param_history, df_history, x_data, y_data = perform_optimization(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)

            residuals_values = result.residual

            return create_histogram(residuals_values)

        except Exception as e:
            return go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))