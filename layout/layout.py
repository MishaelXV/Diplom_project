from dash import dcc, html
from dash import dash_table

# Начальные значения параметров
a_initial = 1
b_values_initial = [200]
A_initial = 1
TG0_initial = 1
atg_initial = 0.0001
sigma_initial = 0.01
zInf = 100000
N_initial = 50

def create_layout():
    """
        Создает HTML-разметку для веб-приложения

        Возвращает
        -------
        html.Div
            Главный контейнер приложения с параметрами, графиками и кнопками

        """
    return html.Div(
    style={
        'backgroundColor': '#f5f5f5',
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
    },
    children=[
        html.Div(
            style={
                'width': '100%',
                'backgroundColor': '#34495e',
                'color': 'white',
                'padding': '15px',
                'textAlign': 'center',
                'fontSize': '1.8em',
                'fontFamily': 'Arial, sans-serif',
            },
            children=[
                html.H1("Активная термометрия"),
            ]
        ),
        dcc.Store(id='boundary-store', data=[]),
        html.Div(
            style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'width': '100%'},
            children=[
                html.Div(
                    style={
                        'width': '30%',
                        'height': '700px',
                        'overflowY': 'scroll',
                        'padding': '15px',
                        'backgroundColor': '#ffffff',
                        'margin': '15px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'border': '1px solid #cccccc',
                    },
                    children=[
                        html.H1(
                            "Параметры",
                            style={
                                'textAlign': 'center',
                                'color': '#34495e',
                                'fontSize': '1.5em',
                                'fontFamily': 'Arial, sans-serif',
                                'borderBottom': '1px solid #cccccc',
                                'paddingBottom': '5px',
                                'backgroundColor': '#ffffff',
                                'padding': '5px',
                                'width': '100%',
                            }
                        ),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Количество изолированных участков скважины:", style={'color': '#34495e'}),
                                dcc.Input(id="a-input", type="number", value=a_initial,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(id='dynamic-b-inputs'),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Параметр A:", style={'color': '#34495e'}),
                                dcc.Input(id="A-input", type="number", value=A_initial,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Параметр TG0:", style={'color': '#34495e'}),
                                dcc.Input(id="TG0-input", type="number", value=TG0_initial, step=0.1,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Параметр atg:", style={'color': '#34495e'}),
                                dcc.Input(id="atg-input", type="number", value=atg_initial, step=0.0001,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Параметр sigma:", style={'color': '#34495e'}),
                                dcc.Input(id="sigma-input", type="number", value=sigma_initial, step=0.01,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(
                            style={'marginBottom': '10px'},
                            children=[
                                html.Label("Количество замеров:", style={'color': '#34495e'}),
                                dcc.Input(id="N-input", type="number", value=N_initial, step=1,
                                          style={'width': '100%', 'fontSize': '1em', 'margin': '5px'}),
                            ]
                        ),
                        html.Div(
                            style={'marginBottom': '10px', 'width': '100%'},
                            children=[
                                html.Label("Расходы:", style={'color': '#34495e'}),
                                html.Div(
                                    id="debits-output",
                                    style={
                                        'width': '100%',
                                        'fontSize': '1em',
                                        'margin': '0 auto',
                                        'padding': '10px',
                                        'backgroundColor': '#f5f5f5',
                                        'border': '1px solid #cccccc',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'boxSizing': 'border-box',
                                    },
                                    children="Нет данных"
                                ),
                                html.Button(
                                    "Вычислить расходы",
                                    id="calculate-debits-btn",
                                    style={
                                        'backgroundColor': '#34495e',
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'margin': '10px auto',
                                        'border': 'none',
                                        'cursor': 'pointer',
                                        'display': 'block',
                                        'textAlign': 'center',
                                    }
                                ),
                                html.Button(
                                    "Решить обратную задачу",
                                    id="solve_inverse_task",
                                    style={
                                        'backgroundColor': '#34495e',
                                        'color': 'white',
                                        'padding': '10px 20px',
                                        'margin': '10px auto',
                                        'border': 'none',
                                        'cursor': 'pointer',
                                        'display': 'block',
                                        'textAlign': 'center',
                                    }
                                ),
                            ]
                        )
                    ]
                ),
                # Контейнер для графика
                html.Div(
                    style={
                        'display': 'flex',
                        'width': '65%',
                        'marginLeft': '15px',
                        'marginRight': '15px',
                        'marginTop': '15px',
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #cccccc',
                        'padding': '15px',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                    },
                    children=[
                        dcc.Graph(id='quadratic-graph'),
                    ]
                ),
            ]
        ),

        html.Div(
    id="animation-container",
    style={
        'display': 'none',
        'width': '100%',
        'height': '400px',
        'border': '1px solid #ccc',
        'padding': '10px',
        'boxSizing': 'border-box',
        'backgroundColor': '#ffffff'
    },
    children=[
        dcc.Graph(id='animation-graph'),

        html.Button(
            '⛶',
            id='fullscreen-button',
            n_clicks=0,
            style={
                'position': 'absolute',
                'top': '980px',
                'right': '155px',
                'zIndex': 1100,
                'border-radius': '50%',
                'width': '40px',
                'height': '40px',
                'border': '1px solid #ccc',
                'background-color': 'white',
                'cursor': 'pointer',
                'text-align': 'center',
                'line-height': '50px',
                'font-size': '24px',
                'color': '#007BFF',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center'
            }
        ),
    ]
),
        html.Div(
            id = "detal-container",
            style={'display': 'none', 'flexDirection': 'row', 'alignItems': 'flex-start', 'width': '100%'},
            children=[
                html.Div(
                    style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'width': '60%'},
                    children=[
                        html.Div(
                            style={
                                'display': 'flex',
                                'marginLeft': '15px',
                                'marginTop': '15px',
                                'backgroundColor': '#ffffff',
                                'border': '1px solid #cccccc',
                                'padding': '15px',
                                'flexDirection': 'column',
                                'alignItems': 'center',
                                'width': '100%',
                            },
                            children=[
                                dcc.Graph(id='parameters-graph'),
                            ]
                        ),
                        html.Div(id = 'residual-container',
                            style={
                                'display': 'flex',
                                'marginLeft': '15px',
                                'marginTop': '15px',
                                'backgroundColor': '#ffffff',
                                'border': '1px solid #cccccc',
                                'padding': '15px',
                                'flexDirection': 'column',
                                'alignItems': 'center',
                                'width': '100%'
                            },
                            children=[
                                dcc.Graph(id='residual-graph'),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    style={
                        'display': 'flex',
                        'width': '35%',
                        'height': '100%',
                        'marginLeft': '60px',
                        'flexDirection': 'column',
                    },
                    children=[
                        html.Div(
                            style={
                                'display': 'flex',
                                'width': '100%',
                                'height': '330px',
                                'marginLeft': '15px',
                                'marginTop': '15px',
                                'backgroundColor': '#ffffff',
                                'border': '1px solid #cccccc',
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
                                    },
                                    style_cell={
                                        'textAlign': 'left',
                                        'height': 'auto',
                                        'overflowX': 'auto',
                                        'lineHeight': '15px',
                                        'whiteSpace': 'normal',
                                    },
                                    style_header={
                                        'whiteSpace': 'normal',
                                    },
                                    page_size=1000,
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'width': '100%',
                                'height': '330px',
                                'marginLeft': '15px',
                                'marginTop': '15px',
                                'backgroundColor': '#ffffff',
                                'border': '1px solid #cccccc',
                                'flexDirection': 'column',
                                'overflowY': 'auto',
                            },
                            children = [
                                dcc.Graph(id = 'hist-graph', config={'displayModeBar': False}),

                            ]
                        )
                    ]
                ),

            ]
        ),
    ]
)

