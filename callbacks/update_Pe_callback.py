from dash import dcc, html
from dash.dependencies import Input, Output

def register_Pe_callback(app):
    @app.callback(
        Output('dynamic-b-inputs', 'children'),
        Input('a-input', 'value')
    )
    def update_b_inputs(num_b):
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