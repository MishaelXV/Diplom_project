import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from calculates_block.calculates import prepare_dataframe_2
from components.graphs import create_iterations_traces, create_update_buttons

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

            num_pe_params = len(boundaries_data['right']) - 1

            fig = create_iterations_traces(df_history, num_pe_params)

            buttons = create_update_buttons(num_pe_params)

            if num_pe_params > 1:
                buttons.append(dict(
                    label="Все параметры",
                    method="update",
                    args=[{"visible": [True] * num_pe_params + [False]},
                          {"title": "Все параметры"}]
                ))

            fig.update_layout(
                height=300,
                width=1000,
                title_text='График параметров',
                title_x=0.5,
                xaxis_title="Итерации",
                yaxis_title="Значения",
                xaxis=dict(
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgray',
                    mirror=True
                ),
                yaxis=dict(
                    showline=True,
                    linecolor='black',
                    gridcolor='lightgray',
                    mirror=True
                ),
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