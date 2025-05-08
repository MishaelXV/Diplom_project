import dash
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from trainer_app.components.graphs import create_residuals_traces, create_update_res_buttons

def register_residuals_callback(app):
    @app.callback(
        Output('residual-graph', 'figure'),
        [Input('optimization-cache', 'data'),
         Input('boundaries-cache', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value')],
        prevent_initial_call=True
    )
    def update_residual_graph(optimization_data, boundaries_data, A, TG0, atg, b_values):
        if not optimization_data or not boundaries_data:
            return dash.no_update

        if len(b_values) == 1:
            return dash.no_update

        try:
            left_boundary = boundaries_data['left']
            right_boundary = boundaries_data['right']
            x_data = np.array(optimization_data['x_data'])
            y_data = np.array(optimization_data['y_data'])
            Pe_opt = optimization_data['found_Pe']

            traces = create_residuals_traces(Pe_opt, x_data, y_data, TG0, atg, A, left_boundary, right_boundary)

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
                          {"title": "E от всех параметров"}],
                )
                buttons.append(button_all)

            colors = {
                'noisy': '#E63946',
                'true': '#A8DADC',
                'geo': '#457B9D',
                'border': '#2A2A2A',
                'text': '#F1FAEE',
                'grid': '#1A1A1A'
            }

            fig.update_layout(
                height=300,
                width=1000,
                title_text='E от параметров',
                title_x=0.5,
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color=colors['text']
                ),
                xaxis=dict(
                    title='Значение параметра',
                    showline=True,
                    linecolor=colors['border'],
                    linewidth=1.5,
                    mirror=False,
                    gridcolor=colors['grid'],
                    griddash='dot',
                    zeroline=False,
                    tickfont=dict(size=15, color=colors['text'])
                ),
                yaxis=dict(
                    title='J',
                    linecolor=colors['border'],
                    linewidth=1.5,
                    mirror=False,
                    gridcolor=colors['grid'],
                    griddash='dot',
                    zeroline=False,
                    tickfont=dict(size=15, color=colors['text'])
                ),
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                margin=dict(l=20, r=20, t=40, b=20),
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons
                    )
                ]
            )

            return fig

        except Exception as e:
            return go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))