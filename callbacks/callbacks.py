import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from block.block import TsGLin, calculate_TsGLin_array, geoterma, debit
from optimizator.optimizer import piecewise_constant, run_optimization, residuals_
import time
import plotly.io as pio

rng = np.random.default_rng(42)
pio.templates.default = "seaborn"

def register_callbacks(app):

    @app.callback(
        Output('dynamic-b-inputs', 'children'),
        Input('a-input', 'value')
    )
    def update_b_inputs(num_b):
        """
       Обновляет Пекле на каждом интервале.

       Параметры
       ----------
       num_b : int
           Общее количество параметров Пекле.

       Возвращает
       -------
       float
           Значения Пекле на каждом интервале.
        """
        a_initial = 1
        b_values_initial = [200]
        inputs = []
        for i in range(num_b):
            inputs.append(
                html.Div(
                    style={'marginBottom': '10px'},
                    children=[
                        html.Label(f"{i + 1} интервал:", style={'color': '#34495e'}),
                        html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                            html.Label("Пекле:", style={'color': '#34495e', 'marginRight': '10px'}),
                            dcc.Input(id={'type': 'b-input', 'index': i}, type="number", value=b_values_initial[0],
                                      placeholder='Введите значение Пекле',
                                      style={'flex': '1', 'fontSize': '1em', 'margin': '5px'}),
                        ]),
                        html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                            html.Label("Граница интервала (вводить через пробел):",
                                       style={'color': '#34495e', 'marginRight': '10px'}),
                            dcc.Input(id={'type': 'boundary-input', 'index': i}, type="text", value="0 0",
                                      placeholder='Введите границы через пробел',
                                      style={'flex': '1', 'fontSize': '1em', 'margin': '5px', 'width': '80%'}),
                            html.Button("Запомнить", id={'type': 'submit-boundary', 'index': i}),
                        ]),
                    ]
                )
            )
        return inputs

    @app.callback(
        Output('boundary-store', 'data'),
        Input({'type': 'submit-boundary', 'index': dash.dependencies.ALL}, 'n_clicks'),
        [State({'type': 'boundary-input', 'index': dash.dependencies.ALL}, 'value')]
    )
    def save_boundaries(n_clicks, boundary_values):
        if n_clicks and any(n_clicks):
            left_boundary = []
            right_boundary = []
            for i, n in enumerate(n_clicks):
                boundaries = boundary_values[i].strip().split()
                if len(boundaries) >= 2:
                    left_boundary.append(float(boundaries[0]))
                    right_boundary.append(float(boundaries[1]))
                else:
                    print(f"Некорректные границы для интервала {i + 1}: {boundary_values[i]}")

            return {'left': left_boundary, 'right': right_boundary}


    def extract_boundaries(boundary_values):
        left_boundary = boundary_values.get('left', [])
        right_boundary = boundary_values.get('right', [])
        left_boundary = [float(b) for b in left_boundary]
        right_boundary = [float(b) for b in right_boundary]
        return left_boundary, right_boundary


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
        """
         Обновляет график решения прямой задачи.

         Параметры
         ----------
         a : int
            Общее количество изолированных интервалов.
         b_values : float
            Значения параметров Пекле на каждом интервале
         boundary_values : float
            Значения границ
         A : float
            Коэффициент теплопередачи.
         TG0 : float
            Начальная температура на поверхности.
         atg : float
             Геотермический градиент (изменение температуры с глубиной).
         sigma : float
             Стандартное отклонение шума в данных.

         Возвращает
         -------
         graph
             Профиль температуры.
        """
        try:
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

            TsGLin_init = 0
            TsGLin_array = calculate_TsGLin_array(right_boundary, 100000, TG0, atg, A, b_values, left_boundary,
                                                  TsGLin_init,
                                                  a)
            TsGLin_array = [float(value) for value in TsGLin_array]
            z_all = []
            T_all = []

            for j in range(a):
                z = np.linspace(left_boundary[j], right_boundary[j], N)
                T_list = [TsGLin([z_val], 100000, TG0, atg, A, b_values[j], left_boundary[j], TsGLin_array[j]) for z_val
                          in z]
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

            # with open('../tests/data/values/temperatura_values.txt', 'w') as file:
            #     for value in T_all:
            #         file.write(f"{value}\n")

            noise = np.random.normal(loc=0, scale=sigma, size=len(T_all))
            T_all_noisy = np.array(T_all) + noise

            fig = go.Figure()

            noisy_trace = go.Scatter(x=z_all, y=T_all_noisy, mode='lines', name='Шум', line=dict(width=3))
            fig.add_trace(noisy_trace)

            temperature_trace = go.Scatter(x=z_all, y=T_all, mode='lines', name='Температура',
                                           line=dict(width=2))
            fig.add_trace(temperature_trace)

            z_values_ = np.linspace(left_boundary[0], right_boundary[a - 1], 200)
            result_values = geoterma(z_values_, TG0, atg)
            fig.add_trace(
                go.Scatter(x=z_values_, y=result_values, mode='lines', name='Геотерма', line=dict(width=2)))

            fig.update_layout(
                xaxis_title=dict(text="z/rw", font=dict(size=28)),
                yaxis_title=dict(text="θ", font=dict(size=28)),
                width=1100,
                height=700,
                title=dict(text="Профиль температуры", font=dict(size=24), x=0.5, xanchor='center'),
                yaxis=dict(gridcolor='lightgray', linecolor='black', mirror = True),
                xaxis=dict(gridcolor='lightgray', linecolor='black', mirror = True),
                plot_bgcolor='white',

            )

            return fig

        except ValueError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except TypeError as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))
        except Exception as e:
            return go.Figure(layout=go.Layout(title="Введите значения"))

    @app.callback(
        Output("debits-output", "children"),
        [
            Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
            Input("calculate-debits-btn", "n_clicks"),
            State('boundary-store', 'data'),
        ]
    )
    def update_debits(pe_values, n_clicks, boundary_values):
        """
         Обновляет значения расходов.

         Параметры
         ----------
         pe_values : float
            Значения параметров Пекле на каждом интервале.
         n_clicks : bool
            Нажатие клавиши вычислить расходы.
         boundary_values : float
            Значения границ.

         Возвращает
         -------
         float
             Значения расходов на каждом проточном участке.
        """
        if not n_clicks:
            return "Нажмите 'Вычислить расходы', чтобы обновить значения."

        if not pe_values or any(pe is None or pe == '' for pe in pe_values):
            return "Расходы не вычислены. Убедитесь, что заданы значения Пекле для всех участков."

        try:
            pe_values = [float(pe) for pe in pe_values]

            if len(pe_values) <= 1:
                return "Общее число участков должно быть больше одного. Проверьте введённые данные."

            if any(pe < 0 for pe in pe_values):
                return "Ошибка: значения Пекле не могут быть отрицательными. Проверьте введённые данные."

            if not boundary_values:
                return "Ошибка: Границы интервалов отсутствуют."

            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])

            if len(left_boundary) != len(pe_values) or len(right_boundary) != len(pe_values):
                return "Ошибка: Количество границ не соответствует числу значений Пекле."

            if all(value == 0 for value in left_boundary) or all(value == 0 for value in right_boundary):
                return "Ошибка: Границы интервалов отсутствуют"

            debits = [debit(pe) for pe in pe_values]

            debits_elements = [
                html.Div(f"Изолированный участок {i + 1}: Расход = {debit_value:.5f} м³/сут")
                for i, debit_value in enumerate(debits[:-1])
            ]

            return debits_elements
        except ValueError:
            return "Ошибка: все значения Пекле должны быть числовыми."
        except Exception as e:
            return f"Ошибка при вычислении дебитов: {e}"


    @app.callback(
        Output('animation-container', 'style'),
        Output('animation-graph', 'figure'),
        Input('solve_inverse_task', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_animation(n_clicks, b_values, boundary_values, A, TG0, atg, sigma, N):
        """
        Обновляет график решения обратной задачи с анимацией.

        Параметры
        ----------
        n_clicks : bool
           Нажатие клавиши решить обратную задачу.
        b_values : float
            Значения Пекле на каждом интервале.
        boundary_values : float
           Значения границ.
        A : float
            Коэффициент теплопередачи.
        TG0 : float
             Начальная температура на поверхности.
         atg : float
             Геотермический градиент (изменение температуры с глубиной).
        sigma : float
            Стандартное отклонение шума в данных.

        Возвращает
        -------
        graph
            График с анимацией.
        """
        if not n_clicks:
            return {'display': 'none'}, dash.no_update
        if len(b_values) == 1:
            return {'display': 'none'}, dash.no_update

        try:
            frames = []
            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])
            left_boundary = [float(b) for b in left_boundary]
            right_boundary = [float(b) for b in right_boundary]

            result, param_history, df_history, x_data, y_data = run_optimization(left_boundary, right_boundary, b_values, 100000, TG0, atg, A, sigma, N, rng=rng)

            for i, (params_dict, _) in enumerate(param_history):
                Pe_values = list(params_dict.values())
                Pe_values.append(0)
                y_predicted = piecewise_constant(params_dict, x_data, 100000, TG0, atg, A, Pe_values, left_boundary, right_boundary)

                frame = go.Frame(
                    data=[
                        go.Scatter(
                            x=x_data,
                            y=y_predicted,
                            mode='lines',
                            name='Подгонка',
                            line=dict(dash='dash', width=3)
                        ),
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            name='Замеры',
                            marker=dict(size=11, opacity=0.5)
                        )

                    ],
                    name=f'Итерация_{i}'
                )
                frames.append(frame)

            fig = go.Figure(
                data = [
                    go.Scatter(
                        x=x_data,
                        y=piecewise_constant(param_history[0][0], x_data, 100000, TG0, atg, A, [1 for _ in range(len(b_values))], left_boundary, right_boundary),
                        mode='lines',
                        name='Подгонка',
                        line=dict(dash='dash', width=3)
                    ),
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        name='Замеры',
                        marker=dict(size=11, opacity=0.5)
                    )
                ],
                layout=go.Layout(
                    title=dict(text="Обратная задача", font=dict(size=24), x=0.5, xanchor='center'),
                    xaxis_title=dict(text="z/rw", font=dict(size=18)),
                    yaxis_title=dict(text="θ", font=dict(size=18)),
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
                    # legend=dict(
                    #     x=0.1,  # Положение легенды на графике по оси x
                    #     y=0.9,  # Положение легенды на графике по оси y
                    #     bgcolor='rgba(255, 255, 255, 0.7)',  # Цвет фона легенды с прозрачностью
                    #     bordercolor='black',  # Цвет границы легенды
                    #     borderwidth=1  # Толщина границы легенды
                    # ),
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=5),
                    updatemenus=[
                        dict(
                            type="buttons",
                            x=0.05,
                            y = -0.2,
                            showactive=False,
                            direction='right',
                            buttons=[
                                dict(
                                    label="▶",
                                    method="animate",
                                    args=[
                                        None,
                                        dict(
                                            frame=dict(duration=300, redraw=True),
                                            fromcurrent=True,
                                            mode='immediate'
                                        )
                                    ]
                                ),
                                dict(
                                    label="■",
                                    method="animate",
                                    args=[
                                        [None],
                                        dict(
                                            frame=dict(duration=0, redraw=False),
                                            mode="immediate"
                                        )
                                    ]
                                )
                            ]
                        )
                    ],

                    sliders=[
                        dict(
                            active=0,
                            currentvalue=dict(
                                font=dict(size=16),
                                prefix="Итерация: ",
                                visible=True,
                                xanchor="left"
                            ),
                            x=0.06,
                            y = -0.005,
                            len=0.95,
                            steps=[
                                dict(
                                    label=i,
                                    method="animate",
                                    args=[
                                        [f'Итерация_{i}'],
                                        dict(
                                            frame=dict(duration=0, redraw=True),
                                            mode="immediate"
                                        )
                                    ]
                                )
                                for i in range(len(frames))
                            ]
                        )
                    ]
                ),
                frames=frames
            )

            return {
                'display': 'block',
                'width': '98.5%',
                'height': '450px',
                'border': '1px solid #ccc',
                'padding': '10px',
                'boxSizing': 'border-box',
                'backgroundColor': '#ffffff'
            }, fig

        except Exception as e:
            return {'display': 'none'}, go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))

    @app.callback(
        Output('table-graph', 'data'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_table(b_values, boundary_values, A, TG0, atg, sigma, N):
        """
        Обновляет таблицу с итерациями и графиками параметров от итераций.

        Параметры
        ----------
        b_values : list
            Значения Пекле на каждом интервале.
        boundary_values : dict
            Значения границ.
        A : float
            Коэффициент теплопередачи.
        TG0 : float
            Начальная температура на поверхности.
        atg : float
            Геотермический градиент (изменение температуры с глубиной).
        sigma : float
            Стандартное отклонение шума в данных.

        Возвращает
        -------
        data : list
            Данных для отображения в таблице.
        """
        try:
            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])
            left_boundary = [float(b) for b in left_boundary]
            right_boundary = [float(b) for b in right_boundary]

            result, param_history, df_history, x_data, y_data = run_optimization(
                left_boundary, right_boundary, b_values, 100000, TG0, atg, A, sigma, N, rng=rng)

            def round_mantissa(value, decimals=5):
                formatted = f"{value:.{decimals}e}"
                return float(formatted)

            df_history['Невязка'] = df_history['Невязка'].apply(round_mantissa)

            df_history = df_history.reset_index()
            df_history.rename(columns={'index': 'Итерация'}, inplace=True)

            return df_history.to_dict('records')

        except Exception as e:
            return [{'Итерация': 'Ошибка', 'Невязка': str(e)}]

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
    def parameters_graph(b_values, boundary_values, A, TG0, atg, sigma, N):
        """
        Обновляет графики параметров от итераций.

        Параметры
        ----------
        b_values : list
            Значения Пекле на каждом интервале.
        boundary_values : dict
            Значения границ.
        A : float
            Коэффициент теплопередачи.
        TG0 : float
            Начальная температура на поверхности.
        atg : float
            Геотермический градиент (изменение температуры с глубиной).
        sigma : float
            Стандартное отклонение шума в данных.

        Возвращает
        -------
        fig : go.Figure
            График с параметрами.
        """
        try:
            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])
            left_boundary = [float(b) for b in left_boundary]
            right_boundary = [float(b) for b in right_boundary]

            result, param_history, df_history, x_data, y_data = run_optimization(
                left_boundary, right_boundary, b_values, 100000, TG0, atg, A, sigma, N, rng=rng
            )

            df_history = df_history.reset_index()
            df_history.rename(columns={'index': 'Итерация'}, inplace=True)

            num_pe_params = len(b_values) - 1

            fig = go.Figure()

            for i in range(num_pe_params):
                trace = go.Scatter(
                    x=df_history.index,
                    y=df_history[f"Pe_{i + 1}"],
                    mode="lines",
                    name=f"Pe_{i + 1}",
                    visible=False,
                )
                fig.add_trace(trace)

            nevyazka_trace = go.Scatter(
                x=df_history.index,
                y=df_history["Невязка"],
                mode="lines",
                name="Невязка",
                visible=False,
            )
            fig.add_trace(nevyazka_trace)

            buttons = []

            for i in range(num_pe_params):
                visible = [False] * (num_pe_params + 1)
                visible[i] = True
                buttons.append(dict(
                    label=f"Pe_{i + 1}",
                    method="update",
                    args=[{"visible": visible + [False]},
                          {"title": f"Pe на {i + 1} интервале"}]
                ))

            buttons.append(dict(
                label="Невязка",
                method="update",
                args=[{"visible": [False] * num_pe_params + [True]},
                      {"title": "Невязка"}]
            ))

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
                # yaxis_range=[-3.5, 1.3],
                # yaxis_type='log',
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

    @app.callback(
        [Output('animation-graph', 'style')],
        [Input('fullscreen-button', 'n_clicks')],
        [State('animation-graph', 'style')]
    )
    def toggle_fullscreen(n_clicks, graph_style):
        if n_clicks % 2 == 1:
            return [{
                'height': '100vh',
                'width': '100vw',
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'zIndex': 1000
            }, ]
        else:
            return [{
                'height': '400px',
                'width': '100%',
                'position': 'relative',
            }]

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
    def update_residual(n_clicks, b_values, boundary_values, A, TG0, atg, sigma, N):
        """
        Обновляет графики невязки от параметров.

        Параметры
        ----------
        n_clicks : bool
           Нажатие клавиши решить обратную задачу.
        b_values : float
            Значения Пекле на каждом интервале.
        boundary_values : float
           Значения границ.
        A : float
            Коэффициент теплопередачи.
        TG0 : float
             Начальная температура на поверхности.
         atg : float
             Геотермический градиент (изменение температуры с глубиной).
        sigma : float
            Стандартное отклонение шума в данных.

        Возвращает
        -------
        graph
            Графики невязки от каждого параметра.
        """
        if not n_clicks:
            return {'display': 'none'}, dash.no_update
        if len(b_values) == 1:
            return {'display': 'none'}, dash.no_update

        try:
            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])
            left_boundary = [float(b) for b in left_boundary]
            right_boundary = [float(b) for b in right_boundary]
            result, param_history, df_history, x_data, y_data = run_optimization(left_boundary, right_boundary, b_values,
                                                                                 100000, TG0, atg, A, sigma, N, rng=rng)
            num_params = len(result.params)

            param_values = [np.linspace(0, 1000, 100) for _ in range(num_params)]

            fixed_params = result.params.valuesdict()

            traces = []

            for i, param_name in enumerate(result.params):
                residuals_param = []
                for param_val in param_values[i]:
                    params_dict = fixed_params.copy()
                    params_dict[param_name] = param_val
                    res = residuals_(params_dict, x_data, y_data, 100000, TG0, atg, A, b_values, left_boundary,
                                     right_boundary)
                    residuals_param.append(np.sum(res ** 2))

                traces.append(go.Scatter(
                    x=param_values[i],
                    y=residuals_param,
                    mode='lines',
                    name=f'Pe_{i + 1}',
                    visible=(i == 0)
                ))

            fig = go.Figure()

            traces_with_visibility = []
            for i, trace in enumerate(traces):
                traces_with_visibility.append(trace)
                traces_with_visibility[-1]['visible'] = False
                fig.add_trace(trace)

            buttons = []

            for i, trace in enumerate(traces_with_visibility):
                visible = [False] * len(traces_with_visibility)
                visible[i] = True
                button = dict(
                    label=trace['name'],
                    method="update",
                    args=[{"visible": visible},
                          {"title": f"Значение невязки от Pe на {i + 1} интервале"}],
                )
                buttons.append(button)

            if len(right_boundary) > 2:
                visible_all = [True] * len(traces_with_visibility)
                button_all = dict(
                    label="Все параметры",
                    method="update",
                    args=[{"visible": visible_all},
                          {"title": "Все параметры"}],
                )
                buttons.append(button_all)

            visible_initial = [False] * len(traces_with_visibility)

            fig.update_layout(
                height=300,
                width=1000,
                title_text='Невязка от параметров',
                title_x=0.5,
                xaxis=dict(title="Значение параметра",
                           showline=True,
                           linecolor = 'black',
                           gridcolor = 'lightgray',
                           mirror = True),
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

    @app.callback(
        Output('hist-graph', 'figure'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_hist(b_values, boundary_values, A, TG0, atg, sigma, N):
        """
        Обновляет гистограмму.

        Параметры
        ----------
        b_values : list
            Значения Пекле на каждом интервале.
        boundary_values : dict
            Значения границ.
        A : float
            Коэффициент теплопередачи.
        TG0 : float
            Начальная температура на поверхности.
        atg : float
            Геотермический градиент (изменение температуры с глубиной).
        sigma : float
            Стандартное отклонение шума в данных.

        Возвращает
        -------
        fig : go.figure
            Данных для отображения в таблице.
        """
        try:
            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])
            left_boundary = [float(b) for b in left_boundary]
            right_boundary = [float(b) for b in right_boundary]

            result, param_history, df_history, x_data, y_data = run_optimization(
                left_boundary, right_boundary, b_values, 100000, TG0, atg, A, sigma, N, rng = rng
            )

            residuals_values = result.residual
            fig = go.Figure(data=[go.Histogram(x=residuals_values)])
            fig.update_layout(
                title="Гистограмма остатков",
                title_x = 0.5,
                margin=dict(l=0, r=0, b=0, t=50, pad=0),
                plot_bgcolor='white',
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
                )
            )

            return fig

        except Exception as e:
            return go.Figure(layout=go.Layout(title=f"Ошибка: {e}"))

    @app.callback(
        Output('detal-container', 'style'),
        Input('solve_inverse_task', 'n_clicks'),
    )
    def update_detal(n_clicks):
        if not n_clicks:
            return {'display': 'none'}
        else:
            time.sleep(8),
            return {
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'flex-start',
                'width': '100%'
            }



