import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from block.cache import perform_optimization_cached, get_boundaries_cached, extract_boundaries_cached
from block.calculates import perform_optimization
from components.boundaries import extract_boundaries
from components.graphs import create_iterations_traces, create_update_buttons
import pandas as pd

def prepare_dataframe(df_history):
    df_history = df_history.reset_index()
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history


def register_iterations_callback(app):
    @app.callback(
        Output('parameters-graph', 'figure'),
        [Input('optimization-cache', 'data'),  # Основной источник данных
         Input('boundaries-cache', 'data')],  # Границы из кэша
        prevent_initial_call=True
    )
    def update_iterations_graph(optimization_data, boundaries_data):
        try:
            # Проверяем наличие данных в кэше
            if not optimization_data or not boundaries_data:
                return go.Figure(layout=go.Layout(title="Ожидание данных..."))

            # Восстанавливаем DataFrame из кэша
            df_history = pd.DataFrame(
                data=optimization_data['df_history']['data'],
                columns=optimization_data['df_history']['columns']
            )

            # Подготавливаем DataFrame (если нужно)
            df_history = prepare_dataframe(df_history)

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