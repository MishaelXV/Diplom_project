import plotly.graph_objects as go
import numpy as np
import pandas as pd
from main_block.main_functions import geoterma
from main_block.main_functions import main_func
from optimizator.optimizer import compute_leakage_profile
from trainer_app.components.support_functions import residuals

def create_figure_direct_task(z_all, T_all, T_all_noisy, left_boundary, right_boundary, TG0, atg, a):
    fig = go.Figure()
    colors = {
        'noisy': '#E63946',
        'true': '#A8DADC',
        'geo': '#457B9D',
        'border': '#2A2A2A',
        'text': '#F1FAEE',
        'grid': '#1A1A1A'
    }

    fig.add_trace(go.Scatter(
        x=z_all,
        y=T_all_noisy,
        mode='lines',
        name='Зашумленный профиль',
        line=dict(width=2.5, color=colors['noisy']),
        hovertemplate='<b>Глубина</b>: %{x:.1f}<br><b>Температура</b>: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=z_all,
        y=T_all,
        mode='lines',
        name='Заданный профиль',
        line=dict(width=2.5, color=colors['true']),
        hovertemplate='<b>Глубина</b>: %{x:.1f}<br><b>Температура</b>: %{y:.2f}<extra></extra>'
    ))

    z_values_ = np.linspace(left_boundary[0], right_boundary[a - 1], 200)
    result_values = geoterma(z_values_, TG0, atg)
    fig.add_trace(go.Scatter(
        x=z_values_,
        y=result_values,
        mode='lines',
        name='Геотерма',
        line=dict(width=2.5, color=colors['geo']),
        hovertemplate='<b>Глубина</b>: %{x:.1f}<br><b>Температура</b>: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        font=dict(
            family="Arial, sans-serif",
            size=16,
            color=colors['text']
        ),
        xaxis=dict(
            title=dict(text="Глубина (z/rw)", font=dict(size=18)),
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
            title=dict(text="Температура (θ)", font=dict(size=18)),
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
            y=0.02,
            xanchor='right',
            yanchor='bottom',
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor=colors['border'],
            borderwidth=1,
            font=dict(size=16),
            orientation='h'
        ),
        title=dict(
            text="Температурный профиль",
            font=dict(size=22, family="Arial"),
            x=0.5,
            xanchor='center',
            y=0.95
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        margin=dict(l=80, r=80, t=80, b=80),
        hoverlabel=dict(
            font_size=16,
            font_family="Arial"
        )
    )

    return fig


def generate_frames(param_history, x_data, x_data_true, y_data_noize, left_boundary, right_boundary,
                    TG0, atg, A, fixed_first_pe, fixed_last_pe):
    colors = {
        'model': '#00BFFF',
        'true': '#A8DADC',
        'leakage': '#FF7F50'
    }

    frames = []
    for i, (params_list, _) in enumerate(param_history):
        Pe_values = [fixed_first_pe] + params_list + [fixed_last_pe]
        y_predicted = main_func(x_data, TG0, atg, A, Pe_values, left_boundary, right_boundary)
        leakage = compute_leakage_profile(x_data, left_boundary, right_boundary, Pe_values)

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x_data,
                    y=y_predicted,
                    mode='lines',
                    name='Модельный профиль',
                    line=dict(color=colors['model'], width=2),
                    yaxis='y1'
                ),
                go.Scatter(
                    x=x_data_true,
                    y=y_data_noize,
                    mode='lines',
                    name='Заданный профиль',
                    line=dict(color=colors['true'], width=2),
                    yaxis='y1'
                ),
                go.Scatter(
                    x=x_data,
                    y=leakage,
                    mode='lines',
                    name='Профиль утечки (ΔPe)',
                    line=dict(color=colors['leakage'], width=2, shape='hv'),
                    fill='tozeroy',
                    yaxis='y2'
                )
            ],
            name=f'Итерация_{i}'
        )
        frames.append(frame)
    return frames


def create_figure_animation(frames, x_data, left_boundary, right_boundary,
                            TG0, atg, A, b_values, x_data_true, y_data_noize):
    colors = {
        'model': '#00BFFF',
        'true': '#A8DADC',
        'leakage': '#FF7F50',
        'text': '#F1FAEE',
        'border': '#2A2A2A',
        'grid': '#1A1A1A',
        'background': '#000000'
    }

    initial_y_predicted = main_func(x_data, TG0, atg, A, [1 for _ in range(len(b_values))],
                                    left_boundary, right_boundary)
    initial_leakage = compute_leakage_profile(x_data, left_boundary, right_boundary,
                                            [1 for _ in range(len(b_values))])

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_data,
                y=initial_y_predicted,
                mode='lines',
                name='Модельный профиль',
                line=dict(color=colors['model'], width=2),
                yaxis='y1'
            ),
            go.Scatter(
                x=x_data_true,
                y=y_data_noize,
                mode='lines',
                name='Заданный профиль',
                line=dict(color=colors['true'], width=2),
                yaxis='y1'
            ),
            go.Scatter(
                x=x_data,
                y=initial_leakage,
                mode='lines',
                name='Профиль утечки (ΔPe)',
                line=dict(color=colors['leakage'], width=2, shape='hv'),
                fill='tozeroy',
                yaxis='y2'
            )
        ],
        layout=go.Layout(
            font=dict(
                family="Arial",
                size=16,
                color=colors['text']
            ),
            title=dict(text="Восстановление температурного профиля", font=dict(size=24), x=0.5, xanchor='center'),
            xaxis=dict(
                title=dict(text="Глубина (z/rw)", font=dict(size=18)),
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
                title=dict(text="Температура (θ)", font=dict(size=18)),
                showline=True,
                linecolor=colors['border'],
                linewidth=1.5,
                mirror=False,
                gridcolor=colors['grid'],
                griddash='dot',
                zeroline=False,
                tickfont=dict(size=15, color=colors['text']),
                domain=[0, 0.85]
            ),
            yaxis2=dict(
                title=dict(text="ΔPe", font=dict(size=18)),
                overlaying='y',
                side='right',
                showline=True,
                linecolor=colors['border'],
                linewidth=1.5,
                gridcolor=colors['grid'],
                griddash='dot',
                zeroline=False,
                tickfont=dict(size=15, color=colors['text']),
                anchor='x',
                range=[0, None]
            ),
            legend=dict(
                x=0.98,
                y=0.02,
                xanchor='right',
                yanchor='bottom',
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor=colors['border'],
                borderwidth=1,
                font=dict(size=16),
                orientation='v'
            ),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            height=400,
            margin=dict(l=60, r=100, t=60, b=100),
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
                    font=dict(size=14, color=colors['text'])
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue=dict(
                        font=dict(size=16, color=colors['text']),
                        prefix="Итерация: ",
                        visible=True,
                        xanchor="left"
                    ),
                    x=0.06,
                    y=-0.15,
                    len=0.95,
                    bgcolor=colors['background'],
                    activebgcolor='white',
                    bordercolor=colors['border'],
                    borderwidth=1,
                    font=dict(color=colors['text']),
                    tickcolor=colors['text'],
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


def build_parallel_coordinates_figure(data):
    df = pd.DataFrame(data)
    colors = df['J']
    num_dims = len(df.columns)

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=colors,
                colorscale='Viridis',
                showscale=True,
                cmin=colors.min(),
                cmax=colors.max(),
                colorbar=dict(
                    title='J',
                    tickfont=dict(color='white'),
                    titlefont=dict(color='white')
                )
            ),
            dimensions=[dict(
                label=col,
                values=df[col],
                range=[df[col].min(), df[col].max()],
            ) for col in df.columns],
            labelfont=dict(color='white'),
            tickfont=dict(color='white'),
            rangefont=dict(color='white'),
        )
    )

    spacing = 1 / (num_dims - 1)
    fig.update_layout(
        autosize=True,
        shapes=[
            dict(
                type='line',
                xref='paper',
                yref='paper',
                x0=i * spacing,
                x1=i * spacing,
                y0=0,
                y1=1,
                line=dict(color='white', width=1)
            )
            for i in range(num_dims)
        ],
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        margin=dict(l=25, r=25, t=40, b=0),
        font=dict(color='white', family='Arial'),
    )

    return fig