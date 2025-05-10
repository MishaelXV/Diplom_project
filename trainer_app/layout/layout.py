from dash import dcc, html
from dash import dash_table

a_initial = 1
b_values_initial = [2000]
A_initial = 5
TG0_initial = 1
tg_initial = 0.0001
sigma_initial = 0.001
zInf = 100000
N_initial = 250

html.Div(style={
    'css': [
        {
            'selector': 'input[type="number"]::-webkit-inner-spin-button',
            'rule': '''
                -webkit-appearance: none;
                background: #your_color url(data:image/png;base64,...) no-repeat center center;
                width: 1em;
                opacity: 1;
            '''
        }
    ]
})


def create_header():
    return html.Div(
        style={
            'width': '100%',
            'backgroundColor': '#1e1e1e',
            'color': '#DDDDDD',
            'padding': '15px 20px',
            'textAlign': 'center',
            'fontSize': '1.7em',
            'fontFamily': 'Arial, sans-serif',
            'marginBottom': '15px',
        },
        children=[
            html.H1(
                "Моделирование температурных полей",
                style={
                    'margin': '0',
                    'padding': '5px 0',
                    'fontSize': '1em',
                    'letterSpacing': '0.5px'
                }
            )
        ]
    )


def create_parameters_input():
    return html.Div(
        style={
            'width': '30%',
            'height': '700px',
            'overflowY': 'scroll',
            'padding': '15px',
            'backgroundColor': '#1e1e1e',
            'margin': '15px',
            'display': 'flex',
            'flexDirection': 'column',
            'border': '1px solid #1e1e1e',
        },
        children=[
            html.H1(
                "Параметры",
                style={
                    'textAlign': 'center',
                    'color': '#DDDDDD',
                    'fontSize': '1.5em',
                    'fontFamily': 'Arial, sans-serif',
                    'borderBottom': '1px solid #cccccc',
                    'paddingBottom': '5px',
                    'backgroundColor': '#1e1e1e',
                    'padding': '5px',
                    'width': '100%',
                }
            ),
            html.Div(
                style={'marginBottom': '10px'},
                children=[
                    html.Label("Количество изолированных участков скважины:", id="label-a-input", style={'color': '#DDDDDD'}),
                    dcc.Input(id="a-input", type="number", value=a_initial,
                              style={
                                  'width': '100%',
                                  'fontSize': '1em',
                                  'margin': '5px',
                                  'backgroundColor': '#1e1e1e',
                                  'color': '#DDDDDD',
                                  'border': '1px solid #555'
                              }),
                    html.Div(id='info-a-input', children='', style={
                        'display': 'none',
                        'position': 'absolute',
                        'left': '100%',
                        'top': '0',
                        'width': '250px',
                        'backgroundColor': '#1e1e1e',
                        'border': '1px solid #555',
                        'padding': '10px',
                        'zIndex': '1000',
                        'color': '#DDDDDD',
                        'boxShadow': '0 0 10px rgba(0,0,0,0.5)',
                        'borderRadius': '5px',
                        'marginLeft': '10px'
                    })
                ]
            ),
            html.Div(id='dynamic-b-inputs'),
            create_input_field("A:", "A-input", A_initial),
            create_input_field("TG0:", "TG0-input", TG0_initial, step=0.1),
            create_input_field("tg:", "atg-input", tg_initial, step=0.0001),
            create_input_field("sigma:", "sigma-input", sigma_initial, step=0.0001),
            create_input_field("N:", "N-input", N_initial, step=1),
            create_debit_calculation_section(),
        ]
    )


def create_input_field(label, input_id, value, step=1):
    return html.Div(
        style={
            'marginBottom': '10px',
            'position': 'relative'
        },
        children=[
            html.Label(
                label,
                style={
                    'color': '#DDDDDD',
                    'cursor': 'pointer',
                    'display': 'inline-block'
                },
                id=f"label-{input_id}",
            ),
            html.Div(
                id=f"info-{input_id}",
                children='',
                style={
                    'display': 'none',
                    'position': 'absolute',
                    'left': '100%',
                    'top': '0',
                    'width': '250px',
                    'backgroundColor': '#1e1e1e',
                    'border': '1px solid #555',
                    'padding': '10px',
                    'zIndex': '1000',
                    'color': '#DDDDDD',
                    'boxShadow': '0 0 10px rgba(0,0,0,0.5)',
                    'borderRadius': '5px',
                    'marginLeft': '10px'
                }
            ),
            dcc.Input(
                id=input_id,
                type="number",
                value=value,
                step=step,
                style={
                    'width': '100%',
                    'fontSize': '1em',
                    'margin': '5px',
                    'backgroundColor': '#1e1e1e',
                    'color': '#DDDDDD',
                    'border': '1px solid #555'
                }
            ),
        ]
    )


def create_debit_calculation_section():
    return html.Div(
        style={'marginBottom': '10px', 'width': '100%'},
        children=[
            html.Label("Расходы:", style={'color': '#DDDDDD'}),
            html.Div(
                id="debits-output",
                style={
                    'width': '100%',
                    'fontSize': '0.8em',
                    'margin': '0 auto',
                    'padding': '10px',
                    'backgroundColor': '#1e1e1e',
                    'border': '1px solid #555',
                    'textAlign': 'center',
                    'boxSizing': 'border-box',
                },
                children="Нажмите 'Вычислить расходы', чтобы обновить значения"
            ),
            html.Button(
                "Вычислить расходы",
                id="calculate-debits-btn",
                style={
                    'width': '100%',
                    'backgroundColor': '#1e1e1e',
                    'color': '#DDDDDD',
                    'padding': '12px 20px',
                    'margin': '10px 0',
                    'cursor': 'pointer',
                    'border': '1px solid #555',
                    'display': 'block',
                    'textAlign': 'center',
                    'fontSize': '0.8em',
                }
            ),
            html.Button(
                "Решить обратную задачу",
                id="open-optimizers",
                style={
                    'width': '100%',
                    'backgroundColor': '#1e1e1e',
                    'color': '#DDDDDD',
                    'padding': '12px 20px',
                    'margin': '10px 0',
                    'border': '1px solid #555',
                    'cursor': 'pointer',
                    'display': 'block',
                    'textAlign': 'center',
                    'fontSize': '0.8em',
                }
            ),
            html.Div(
                id='optimizer-method-container',
                style={'display': 'none', 'marginTop': '10px', 'width': '100%'},
                children=[
                    html.Label("Метод оптимизации:", style={'color': '#DDDDDD'}),
                    dcc.Dropdown(
                        id='optimizer-method',
                        options=[
                            {'label': 'Nelder-Mead', 'value': 'nelder-mead'},
                            {'label': 'BFGS', 'value': 'bfgs'},
                            {'label': 'Powell', 'value': 'powell'},
                            {'label': 'L-BFGS-B', 'value': 'l-bfgs-b'},
                            {'label': 'Levenberg-Marquardt', 'value': 'leastsq'},
                            {'label': 'Differential evolution', 'value': 'differential_evolution'},
                            {'label': 'Cobyla', 'value': 'cobyla'},
                            {'label': 'Bayes', 'value': 'bayes'},
                        ],
                        value=None,
                        clearable=False,
                        style={
                            'backgroundColor': '#1e1e1e',
                            'color': '#DDDDDD',
                            'marginBottom': '10px'
                        },
                        className='custom-dropdown'
                    ),
                    html.Div(
                        id='bayes-params-container',
                        style={'display': 'none', 'marginBottom': '10px'},
                        children=[
                            html.Label("Количество итераций:", style={'color': '#DDDDDD'}),
                            dcc.Input(
                                id='bayes-iterations',
                                type='number',
                                min=1,
                                value=100,
                                style={
                                    'width': '100%',
                                    'backgroundColor': '#1e1e1e',
                                    'color': '#DDDDDD',
                                    'border': '1px solid #555'
                                }
                            )
                        ]
                    ),
                    html.Button(
                        "Запустить оптимизацию",
                        id="run-optimization-btn",
                        style={
                            'width': '100%',
                            'backgroundColor': '#1e1e1e',
                            'color': '#DDDDDD',
                            'padding': '12px 20px',
                            'margin': '10px 0',
                            'border': '1px solid #555',
                            'cursor': 'pointer',
                            'display': 'block',
                            'textAlign': 'center',
                            'fontSize': '0.8em',
                        }
                    )
                ]
            ),
            html.Button(
                "Сгенерировать отчёт",
                id="export_to_pdf",
                style={
                    'width': '100%',
                    'backgroundColor': '#1e1e1e',
                    'color': '#DDDDDD',
                    'padding': '12px 20px',
                    'margin': '10px 0',
                    'border': '1px solid #555',
                    'cursor': 'pointer',
                    'display': 'block',
                    'textAlign': 'center',
                    'fontSize': '0.8em',
                }
            ),
        ]
    )


def create_plot_area():
    return html.Div(
        id="plot-area-container",
        style={
            'display': 'none',
            'width': '65%',
            'height': '725px',
            'marginLeft': '15px',
            'marginRight': '15px',
            'marginTop': '15px',
            'backgroundColor': '#111111',
            'border': '1px solid #444444',
            'padding': '0',
            'flexDirection': 'column',
            'alignItems': 'stretch',
        },
        children=[
            dcc.Graph(
                id='quadratic-graph',
                style={
                    'width': '100%',
                    'height': '100%',
                    'display': 'block'
                },
                config={
                    'responsive': True,
                    'displayModeBar': False,
                }
            )
        ]
    )


def create_animation_container():
    return html.Div(
        id="animation-container",
        style={
            'display': 'none',
            'width': '100%',
            'height': '450px',
            'border': '1px solid #444444',
            'padding': '0',
            'boxSizing': 'border-box',
            'backgroundColor': '#111111',
            'overflow': 'hidden',
            'position': 'relative'
        },
        children=[
            dcc.Loading(
                id="animation-loading",
                type="circle",
                color="#00cc96",
                children=dcc.Graph(
                    id='animation-graph',
                    style={
                        'width': '100%',
                        'height': '100%',
                        'display': 'block'
                    },
                    config={
                        'responsive': True,
                        'displayModeBar': False
                    }
                )
            )
        ]
    )


def create_parallel_coordinates_graph():
    return html.Div(
        style={
            'display': 'flex',
            'margin': '15px',
            'backgroundColor': '#000000',
            'border': '1px solid #444444',
            'padding': '0',
            'flexDirection': 'column',
            'width': 'calc(100% - 30px)',
            'height': '500px',
            'overflow': 'hidden',
        },
        children=[
            dcc.Loading(
                id='parallel-loading',
                type="circle",
                color="#00cc96",
                children=[
                    dcc.Graph(
                        id='parallel-coordinates-graph',
                        config={
                            'responsive': True,
                            'displayModeBar': False,
                        },
                        style={
                            'width': '100%',
                            'height': '100%',
                            'margin': '0',
                            'padding': '0',
                        },
                        figure={
                            'layout': {
                                'margin': {'l': 50, 'r': 50, 't': 30, 'b': 50},
                                'plot_bgcolor': '#000000',
                                'paper_bgcolor': '#000000',
                            }
                        }
                    )
                ],
                style={
                    'width': '100%',
                    'height': '100%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                }
            )
        ]
    )


def create_details_container():
    return html.Div(
        id="detal-container",
        style={'display': 'none', 'flexDirection': 'column', 'alignItems': 'flex-start', 'width': '100%'},
        children=[
            html.Div(
                style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'},
                children=[
                    create_parameters_graph(),
                    create_residuals_graphs(),
                ]
            ),
            create_parallel_coordinates_graph()
        ]
    )


def create_parameters_graph():
    return html.Div(
        style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'width': '60%'},
        children=[
            html.Div(
                style={
                    'display': 'flex',
                    'marginLeft': '15px',
                    'marginTop': '15px',
                    'backgroundColor': '#000000',
                    'border': '1px solid #444444',
                    'padding': '15px',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'width': '100%',
                },
                children=[
                    dcc.Graph(id='parameters-graph', config={
                        'responsive': True,
                        'displayModeBar': False
                        }),

                ]
            ),
            html.Div(
                style={
                    'display': 'flex',
                    'marginLeft': '15px',
                    'marginTop': '15px',
                    'backgroundColor': '#000000',
                    'border': '1px solid #444444',
                    'padding': '15px',
                    'flexDirection': 'column',
                    'alignItems': 'center',
                    'width': '100%'
                },
                children=[
                    dcc.Graph(id='residual-graph', config={
                        'responsive': True,
                        'displayModeBar': False
                        }),
                ]
            ),
        ]
    )


def create_residuals_graphs():
    return html.Div(
        style={
            'display': 'flex',
            'width': '35%',
            'height': '100%',
            'marginLeft': '60px',
            'flexDirection': 'column',
        },
        children=[
            create_data_table(),
            create_optimization_metrics(),
        ]
    )


def create_data_table():
    return html.Div(
        style={
            'display': 'flex',
            'width': '100%',
            'height': '330px',
            'marginLeft': '15px',
            'marginTop': '15px',
            'backgroundColor': '#121212',
            'border': '1px solid #333333',
            'padding': '0',
            'flexDirection': 'column',
            'overflowY': 'auto',
            'overflowX': 'auto',
        },
        children=[
            dash_table.DataTable(
                id='table-graph',
                style_table={
                    'height': '330px',
                    'width': '100%',
                    'margin': '0',
                    'padding': '0',
                    'border': 'none',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'backgroundColor': '#121212'
                },
                style_cell={
                    'textAlign': 'left',
                    'height': 'auto',
                    'overflowX': 'auto',
                    'lineHeight': '18px',
                    'whiteSpace': 'normal',
                    'backgroundColor': '#1e1e1e',
                    'color': '#f1f1f1',
                    'border': '1px solid #333333',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '14px'
                },
                style_header={
                    'backgroundColor': '#2a2a2a',
                    'color': '#ffffff',
                    'fontWeight': 'bold',
                    'fontFamily': 'Consolas, monospace',
                    'fontSize': '15px',
                    'border': '1px solid #444444'
                },
                page_size=1000
            )
        ]
    )


def create_optimization_metrics():
    return html.Div(
        style={
            'display': 'flex',
            'width': '100%',
            'maxWidth': '100%',
            'height': '330px',
            'marginLeft': '15px',
            'marginTop': '15px',
            'backgroundColor': '#000000',
            'border': '1px solid #444444',
            'flexDirection': 'column',
            'overflowY': 'auto',
            'overflowX': 'hidden',
            'padding': '15px',
            'color': '#DDDDDD',
            'fontFamily': 'Arial, sans-serif',
            'boxSizing': 'border-box'
        },
        children=[
            html.H3("Детали оптимизации", style={
                'textAlign': 'center',
                'borderBottom': '1px solid #444',
                'paddingBottom': '10px',
                'marginBottom': '15px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            }),
            html.Div(
                id='optimization-metrics-content',
                style={
                    'overflowY': 'auto',
                    'overflowX': 'hidden',
                    'width': '100%'
                },
                children=[
                    html.Div("Здесь будут отображаться метрики оптимизации", style={
                        'textAlign': 'center',
                        'marginTop': '50px',
                        'color': 'white'
                    })
                ]
            )
        ]
    )


def create_layout():
    return html.Div(
        style={
            'backgroundColor': '#111111',
            'minHeight': '100vh',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
        },
        children=[
            create_header(),
            dcc.Store(id='boundary-store', data=[]),
            dcc.Store(id='optimization-cache'),
            dcc.Store(id='boundaries-cache'),
            html.Div(
                style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'width': '100%'},
                children=[
                    create_parameters_input(),
                    create_plot_area(),
                ]
            ),
            dcc.Store(id='animation-loaded', data=False),
            create_animation_container(),
            create_details_container(),
            dcc.Download(id="download-pdf"),
            dcc.Store(id='stored-figures'),
        ]
    )