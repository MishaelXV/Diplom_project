import dash
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from block.block import calculate_TsGLin_array
from callbacks.graphs import create_figure_direct_task
from callbacks.valid_inputs_of_params import validate_inputs
from block.calculates import calculate_temperatures

def register_direct_task_callback(app):

    def save_temperature_values(T_all, file_path):
        with open(file_path, 'w') as file:
            for value in T_all:
                file.write(f"{value}\n")


    @app.callback(
        Output('quadratic-graph', 'figure'),
        [Input('a-input', 'value'),
         Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_graph(a, b_values, boundary_values, A, TG0, atg, sigma, N):
        try:
            b_values, left_boundary, right_boundary = validate_inputs(a, b_values, boundary_values, A, TG0, atg, sigma)

            TsGLin_init = 0
            TsGLin_array = calculate_TsGLin_array(right_boundary, 100000, TG0, atg, A, b_values, left_boundary, TsGLin_init,
                                                  a)
            TsGLin_array = [float(value) for value in TsGLin_array]

            z_all, T_all = calculate_temperatures(a, left_boundary, right_boundary, N, TG0, atg, A, b_values,
                                                  TsGLin_array)

            noise = np.random.normal(loc=0, scale=sigma, size=len(T_all))
            T_all_noisy = np.array(T_all) + noise

            # save_temperature_values(T_all, '../tests/data/values/temperatura_values.txt')

            fig = create_figure_direct_task(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a)

            return fig

        except ValueError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except TypeError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except Exception as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))