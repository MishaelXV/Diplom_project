import plotly.graph_objects as go
import numpy as np
from main_block.main_functions import geoterma
from main_block.main_functions import main_func
from trainer_app.components.support_functions import residuals

def create_figure_direct_task(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a):
    fig = go.Figure()

    noisy_trace = go.Scatter(
        x=z_all,
        y=T_all_noisy,
        mode='lines',
        name='Зашумленная температура',
        line=dict(width=2, color='#1F77B4')
    )
    fig.add_trace(noisy_trace)

    temperature_trace = go.Scatter(
        x=z_all,
        y=T_all,
        mode='lines',
        name='Истинная температура',
        line=dict(width=2, color='#FF7F0E')
    )
    fig.add_trace(temperature_trace)

    z_values_ = np.linspace(left_boundary[0], right_boundary[a - 1], 200)
    result_values = geoterma(z_values_, TG0, atg)
    fig.add_trace(go.Scatter(
        x=z_values_,
        y=result_values,
        mode='lines',
        name='Геотермический профиль',
        line=dict(width=2, color='#2CA02C')
    ))

    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=14,
            color="black"
        ),

        xaxis_title=dict(text="z/rw", font=dict(size=28)),
        yaxis_title=dict(text="θ", font=dict(size=28)),
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
            x=0.98,
            y=0.02,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0.5)',
            borderwidth=1,
            font=dict(size=16)
        ),

        title=dict(
            text="Профиль температуры",
            font=dict(size=28),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        width=1100,
        height=700,
        margin=dict(l=100, r=100, t=100, b=100)
    )

    return fig


def generate_frames(param_history, x_data, x_data_true, y_data_noize, left_boundary, right_boundary, TG0, atg, A, fixed_first_pe, fixed_last_pe):
    frames = []
    for i, (params_list, _) in enumerate(param_history):
        Pe_values = [fixed_first_pe] + params_list + [fixed_last_pe]
        y_predicted = main_func(x_data, TG0, atg, A, Pe_values, left_boundary, right_boundary)

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x_data,
                    y=y_predicted,
                    mode='lines',
                    name='Модельный профиль',
                    line=dict(color='blue', width=2)
                ),
                go.Scatter(
                    x=x_data_true,
                    y=y_data_noize,
                    mode='lines',
                    name='Истинная температура',
                    line=dict(color='green', width=2)
                )
            ],
            name=f'Итерация_{i}'
        )
        frames.append(frame)
    return frames


def create_figure_animation(frames, x_data, left_boundary, right_boundary, TG0, atg, A, b_values, x_data_true, y_data_noize):
    initial_y_predicted = main_func(x_data, TG0, atg, A, [1 for _ in range(len(b_values))],
                                    left_boundary, right_boundary)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_data,
                y=initial_y_predicted,
                mode='lines',
                name='Модельный профиль',
                line=dict(color='blue', width=2)
            ),
            go.Scatter(
                x=x_data_true,
                y=y_data_noize,
                mode='lines',
                name='Истинная температура',
                line=dict(color='green', width=2)
            )
        ],
        layout=go.Layout(
            font=dict(
                family="Times New Roman",
                size=16,
                color="black"
            ),
            title=dict(text="Обратная задача", font=dict(size=24), x=0.5, xanchor='center'),
            xaxis_title=dict(text="z/rw", font=dict(size=20)),
            yaxis_title=dict(text="θ", font=dict(size=20)),
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
                x=0.989,
                y=0.055,
                xanchor='right',
                yanchor='bottom',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.5)',
                borderwidth=1,
                font=dict(size=16)
            ),
            plot_bgcolor='white',
            height=400,
            margin=dict(l=60, r=20, t=60, b=100),
            updatemenus=[
                dict(
                    type="buttons",
                    x=0.05,
                    y=-0.3,
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
                    ],
                    pad=dict(r=10, t=10, b=10),
                    font=dict(size=14)
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
                    y=-0.15,
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
            y=df_history[f"Pe_{i+1}"],
            mode="lines",
            name=f"Pe_{i + 2}",
            visible=False,
        )
        fig.add_trace(trace)

    # График Невязки (J)
    nevyazka_trace = go.Scatter(
        x=df_history.index,
        y=df_history["Невязка"],
        mode="lines",
        name="J",
        visible=False,
    )
    fig.add_trace(nevyazka_trace)

    error_trace = go.Scatter(
        x=df_history.index,
        y=df_history["residuals"],
        mode="lines",
        name="E",
        visible=False,
    )
    fig.add_trace(error_trace)

    return fig


def create_update_buttons(num_pe_params):
    buttons = []
    total_traces = num_pe_params + 2

    for i in range(num_pe_params):
        visible = [False] * total_traces
        visible[i] = True
        buttons.append(dict(
            label=f"Pe_{i + 2}",
            method="update",
            args=[
                {"visible": visible},
                {
                    "title": f"Pe на {i + 2} интервале",
                    "showlegend": False
                }
            ]
        ))

    # Кнопка J (невязка)
    visible = [False] * total_traces
    visible[num_pe_params] = True
    buttons.append(dict(
        label="J",
        method="update",
        args=[
            {"visible": visible},
            {
                "title": "J",
                "showlegend": False
            }
        ]
    ))

    # Кнопка E (residuals)
    visible = [False] * total_traces
    visible[num_pe_params + 1] = True
    buttons.append(dict(
        label="E",
        method="update",
        args=[
            {"visible": visible},
            {
                "title": "E от итераций",
                "showlegend": False
            }
        ]
    ))

    # Кнопка для всех параметров (Pe)
    visible = [True] * num_pe_params + [False, False]
    buttons.append(dict(
        label="Все параметры",
        method="update",
        args=[
            {"visible": visible},
            {
                "title": "Все параметры",
                "showlegend": True,
                "legend": dict(
                    bgcolor='rgba(0,0,0,0)',
                    borderwidth=0
                )
            }
        ]
    ))

    return buttons


def create_residuals_traces(Pe_opt, x_data, y_data, TG0, atg, A, left_boundary, right_boundary):
    num_params = len(Pe_opt) - 2
    traces = []

    for i in range(num_params):
        param_range = np.linspace(0, 5000, 50)
        residuals_param = []

        for val in param_range:
            Pe_trial = Pe_opt.copy()
            Pe_trial[i + 1] = val
            res = residuals(x_data, y_data, TG0, atg, A, Pe_trial, left_boundary, right_boundary)
            residuals_param.append(np.sum(res ** 2))

        traces.append(go.Scatter(
            x=param_range,
            y=residuals_param,
            mode='lines',
            name=f'Pe_{i + 2}',
            visible=(i == 0)
        ))

    return traces


def create_update_res_buttons(traces):
    buttons = []

    for i, trace in enumerate(traces):
        visible = [False] * len(traces)
        visible[i] = True
        button = dict(
            label=f"Pe_{i + 2}",
            method="update",
            args=[{"visible": visible},
                  {"title": f"E от Pe на {i + 2} интервале"}],
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