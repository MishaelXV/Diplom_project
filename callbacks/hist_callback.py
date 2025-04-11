import plotly.graph_objects as go
from dash.dependencies import Input, Output
from components.graphs import create_histogram

def register_hist_callback(app):
    @app.callback(
        Output('hist-graph', 'figure'),
        Input('optimization-cache', 'data'),
        prevent_initial_call=True
    )
    def update_hist(optimization_data):
        try:
            if not optimization_data:
                raise ValueError("Нет данных в кэше оптимизации")

            residual = optimization_data.get('residual')

            if residual is None:
                residuals_values = [item['residual'] for item in optimization_data['param_history']]
            else:
                residuals_values = residual

            return create_histogram(residuals_values)

        except Exception as e:
            return go.Figure(
                layout=go.Layout(
                    title=f"Ошибка: {str(e)}",
                    xaxis={'visible': False},
                    yaxis={'visible': False}
                )
            )