import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from block.block import TsGLin, calculate_TsGLin_array, geoterma, debit

def register_direct_task_callback(app):
    def extract_boundaries(boundary_values):
        left_boundary = boundary_values.get('left', [])
        right_boundary = boundary_values.get('right', [])
        left_boundary = [float(b) for b in left_boundary]
        right_boundary = [float(b) for b in right_boundary]
        return left_boundary, right_boundary


    def validate_inputs(a, b_values, boundary_values, A, TG0, atg, sigma):
        if any(param is None or param < 0 for param in [a, A, TG0, atg, sigma]):
            raise ValueError("Параметры должны быть заданы и неотрицательными")

        if isinstance(b_values, list):
            b_values = [float(b) for b in b_values]
        else:
            b_values = [float(b_values)]

        if any(b < 0 for b in b_values):
            raise ValueError("Значения Pe должны быть неотрицательными")

        if not boundary_values:
            raise ValueError("Границы должны быть заданы")

        left_boundary, right_boundary = extract_boundaries(boundary_values)

        if any(boundary < 0 for boundary in left_boundary + right_boundary):
            raise ValueError("Границы должны быть неотрицательными")

        if len(b_values) != a:
            raise ValueError("Количество значений b не соответствует количеству интервалов")

        return b_values, left_boundary, right_boundary


    def calculate_temperatures(a, left_boundary, right_boundary, N, TG0, atg, A, b_values, TsGLin_array):
        z_all = []
        T_all = []

        for j in range(a):
            z = np.linspace(left_boundary[j], right_boundary[j], N)
            T_list = [TsGLin([z_val], 100000, TG0, atg, A, b_values[j], left_boundary[j], TsGLin_array[j]) for z_val in
                      z]
            T = [item[0] for item in T_list]

            z_all.extend(z)
            T_all.extend(T)

            if j < a - 1:
                y_value = TsGLin_array[j + 1]
                start_point = right_boundary[j] + 1e-6
                end_point = left_boundary[j + 1]
                z_horizontal = np.linspace(start_point, end_point, N)
                T_all.extend([y_value] * len(z_horizontal))
                z_all.extend(z_horizontal)

        return z_all, T_all


    def create_figure(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a):
        fig = go.Figure()

        noisy_trace = go.Scatter(x=z_all, y=T_all_noisy, mode='lines', name='Шум', line=dict(width=3))
        fig.add_trace(noisy_trace)

        temperature_trace = go.Scatter(x=z_all, y=T_all, mode='lines', name='Температура', line=dict(width=2))
        fig.add_trace(temperature_trace)

        z_values_ = np.linspace(left_boundary[0], right_boundary[a - 1], 200)
        result_values = geoterma(z_values_, TG0, atg)
        fig.add_trace(go.Scatter(x=z_values_, y=result_values, mode='lines', name='Геотерма', line=dict(width=2)))

        fig.update_layout(
            xaxis_title=dict(text="z/rw", font=dict(size=28)),
            yaxis_title=dict(text="θ", font=dict(size=28)),
            width=1100,
            height=700,
            title=dict(text="Профиль температуры", font=dict(size=24), x=0.5, xanchor='center'),
            yaxis=dict(gridcolor='lightgray', linecolor='black', mirror=True),
            xaxis=dict(gridcolor='lightgray', linecolor='black', mirror=True),
            plot_bgcolor='white',
        )

        return fig


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

            fig = create_figure(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a)

            return fig

        except ValueError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except TypeError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except Exception as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))