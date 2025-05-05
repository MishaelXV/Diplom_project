import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from trainer_app.components.support_functions import prepare_dataframe_2
from trainer_app.components.graphs import create_iterations_traces, create_update_buttons

def register_iterations_callback(app):
    @app.callback(
        Output('parameters-graph', 'figure'),
        [Input('optimization-cache', 'data'),
         Input('boundaries-cache', 'data')],
        prevent_initial_call=True
    )
    def update_iterations_graph(optimization_data, boundaries_data):
        try:
            if not optimization_data or not boundaries_data:
                return go.Figure(layout=go.Layout(title="Ожидание данных..."))

            df_history = pd.DataFrame(
                data=optimization_data['df_history']['data'],
                columns=optimization_data['df_history']['columns']
            )

            df_history = prepare_dataframe_2(df_history)

            num_pe_params = len(optimization_data['found_Pe']) - 2

            fig = create_iterations_traces(df_history, num_pe_params)

            buttons = create_update_buttons(num_pe_params)

            fig.update_layout(
                height=300,
                width=1000,
                title_text='График параметров',
                title_x=0.5,
                xaxis_title="Итерации",
                yaxis_title="Значения",
                font=dict(
                    family="Times New Roman",
                    size=14,
                    color="black"
                ),
                xaxis=dict(
                    showline=True,
                    linecolor='black',
                    mirror=True,
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    griddash='dot',
                    gridwidth=0.5
                ),
                yaxis=dict(
                    showline=True,
                    linecolor='black',
                    mirror=True,
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    griddash='dot',
                    gridwidth=0.5
                ),
                legend=dict(
                    x=1,
                    y=1,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=12)
                ),
                showlegend=True,
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons
                    )
                ],
            )

            return fig

        except Exception as e:
            return go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))