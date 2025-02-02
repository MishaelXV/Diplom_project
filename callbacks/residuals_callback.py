import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from block.calculates import perform_optimization
from components.boundaries import extract_boundaries
from components.graphs import create_residuals_traces, create_update_res_buttons

def register_residuals_callback(app):
    @app.callback(
        Output('residual-container', 'style'),
        Output('residual-graph', 'figure'),
        Input('solve_inverse_task', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_residual_graph(n_clicks, b_values, boundary_values, A, TG0, atg, sigma, N):
        if not n_clicks:
            return {'display': 'none'}, dash.no_update
        if len(b_values) == 1:
            return {'display': 'none'}, dash.no_update

        try:
            left_boundary, right_boundary = extract_boundaries(boundary_values)
            result, param_history, df_history, x_data, y_data = perform_optimization(left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)

            traces = create_residuals_traces(result, x_data, y_data, TG0, atg, A, b_values, left_boundary,
                                             right_boundary)

            fig = go.Figure()
            for trace in traces:
                fig.add_trace(trace)

            buttons = create_update_res_buttons(traces)

            if len(right_boundary) > 2:
                visible_all = [True] * len(traces)
                button_all = dict(
                    label="Все параметры",
                    method="update",
                    args=[{"visible": visible_all},
                          {"title": "Все параметры"}],
                )
                buttons.append(button_all)

            fig.update_layout(
                height=300,
                width=1000,
                title_text='Невязка от параметров',
                title_x=0.5,
                xaxis=dict(title="Значение параметра",
                           showline=True,
                           linecolor='black',
                           gridcolor='lightgray',
                           mirror=True),
                yaxis=dict(title="Невязка",
                           showline=True,
                           linecolor='black',
                           gridcolor='lightgray',
                           mirror=True),
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons
                    )
                ]
            )

            return {
                'display': 'flex',
                'marginLeft': '15px',
                'marginRight': '15px',
                'marginTop': '15px',
                'backgroundColor': '#ffffff',
                'border': '1px solid #cccccc',
                'padding': '15px',
                'flexDirection': 'column',
                'alignItems': 'center',
                'width': '100%'
            }, fig

        except Exception as e:
            return {'display': 'none'}, go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))