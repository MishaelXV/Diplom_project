import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from trainer_app.components.support_functions import prepare_dataframe_2
from trainer_app.components.graphs import create_iterations_traces, create_update_buttons

def register_iterations_callback(app):
    @app.callback(
        Output('parameters-graph', 'figure'),
        Output('stored-figures', 'data'),
        [Input('optimization-cache', 'data'),
         Input('boundaries-cache', 'data')],
        prevent_initial_call=True
    )
    def update_iterations_graph(optimization_data, boundaries_data):
        try:
            if not optimization_data or not boundaries_data:
                fig = go.Figure(layout=go.Layout(title="Ожидание данных..."))
                return fig, fig.to_plotly_json()

            df_history = pd.DataFrame(
                data=optimization_data['df_history']['data'],
                columns=optimization_data['df_history']['columns']
            )

            df_history = prepare_dataframe_2(df_history)

            num_pe_params = len(optimization_data['found_Pe']) - 2

            fig = create_iterations_traces(df_history, num_pe_params)

            buttons = create_update_buttons(num_pe_params)

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
                title_text='График параметров',
                title_x=0.5,
                xaxis_title="Итерация",
                yaxis_title="Значения",
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color="white"
                ),
                xaxis=dict(
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
                    showline=True,
                    linecolor=colors['border'],
                    linewidth=1.5,
                    mirror=False,
                    gridcolor=colors['grid'],
                    griddash='dot',
                    zeroline=False,
                    tickfont=dict(size=15, color=colors['text'])
                ),
                legend=dict(
                    x=0.98,
                    y=0.98,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(0, 0, 0, 0.5)',
                    bordercolor=colors['border'],
                    borderwidth=1,
                    font=dict(size=14, color=colors['text']),
                    orientation='v'
                ),
                showlegend=True,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                margin=dict(l=20, r=20, t=40, b=20),
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                    )
                ],
            )

            return fig, fig.to_plotly_json()

        except Exception as e:
            error_fig = go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))
            return error_fig, error_fig.to_plotly_json()