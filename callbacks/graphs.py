import plotly.graph_objects as go
import numpy as np
from block.block import geoterma
from optimizator.optimizer import piecewise_constant, residuals_

def create_figure_direct_task(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a):
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


def generate_frames(param_history, x_data, y_data, left_boundary, right_boundary, TG0, atg, A, b_values):
    frames = []
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
    return frames


def create_figure_animation(frames, x_data, param_history, left_boundary, right_boundary, TG0, atg, A, b_values, y_data):
    initial_params = param_history[0][0] if param_history else {}
    initial_y_predicted = piecewise_constant(initial_params, x_data, 100000, TG0, atg, A,
                                              [1 for _ in range(len(b_values))], left_boundary, right_boundary)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_data,
                y=initial_y_predicted,
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
            height=300,
            margin=dict(l=20, r=20, t=40, b=5),
            updatemenus=[
                dict(
                    type="buttons",
                    x=0.05,
                    y=-0.2,
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
                    y=-0.005,
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
    return fig


def create_iterations_traces(df_history, num_pe_params):
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

    return fig


def create_update_buttons(num_pe_params):
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

    return buttons


def create_residuals_traces(result, x_data, y_data, TG0, atg, A, b_values, left_boundary, right_boundary):
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

    return traces


def create_update_res_buttons(traces):
    buttons = []

    for i, trace in enumerate(traces):
        visible = [False] * len(traces)
        visible[i] = True
        button = dict(
            label=trace['name'],
            method="update",
            args=[{"visible": visible},
                  {"title": f"Значение невязки от Pe на {i + 1} интервале"}],
        )
        buttons.append(button)

    return buttons


def create_histogram(residuals_values):
    fig = go.Figure(data=[go.Histogram(x=residuals_values)])
    fig.update_layout(
        title="Гистограмма отклонений Pe",
        title_x=0.5,
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