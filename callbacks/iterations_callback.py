import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from block.calculates import perform_optimization
from components.boundaries import extract_boundaries
from components.graphs import create_iterations_traces, create_update_buttons

def prepare_dataframe(df_history):
    df_history = df_history.reset_index()
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history


def register_iterations_callback(app):
    @app.callback(
        Output('parameters-graph', 'figure'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_iterations_graph(b_values, boundary_values, A, TG0, atg, sigma, N):
        try:
            left_boundary, right_boundary = extract_boundaries(boundary_values)

            result, param_history, df_history, x_data, y_data = perform_optimization(
                left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)

            df_history = prepare_dataframe(df_history)

            num_pe_params = len(b_values) - 1

            fig = create_iterations_traces(df_history, num_pe_params)

            buttons = create_update_buttons(num_pe_params)

            if len(right_boundary) > 2:
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