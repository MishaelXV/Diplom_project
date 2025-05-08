from dash import dcc, html
from dash.dependencies import Input, Output

def register_Pe_callback(app):
    @app.callback(
        Output('dynamic-b-inputs', 'children'),
        Input('a-input', 'value')
    )
    def update_b_inputs(num_b):
        start_value = 2000
        step = 500
        b_values_initial = [start_value - step * i for i in range(num_b)]

        def generate_boundaries(index):
            base = index * 150
            return f"{base} {base + 100}"

        inputs = []
        for i in range(num_b):
            auto_boundary = generate_boundaries(i)

            inputs.append(
                html.Div(
                    style={'marginBottom': '10px'},
                    children=[
                        html.Label(f"{i + 1} интервал:", style={'color': '#DDDDDD'}),
                        html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                            html.Label("Пекле:", style={'color': '#DDDDDD', 'marginRight': '10px'}),
                            dcc.Input(
                                id={'type': 'b-input', 'index': i},
                                type="number",
                                value=b_values_initial[i],
                                style={
                                    'flex': '1', 'fontSize': '1em', 'margin': '5px',
                                    'backgroundColor': '#1e1e1e', 'color': '#DDDDDD',
                                    'border': '1px solid #555'
                                }
                            ),
                        ]),
                        html.Div(style={'display': 'flex', 'alignItems': 'center'}, children=[
                            html.Label("Граница интервала:", style={'color': '#DDDDDD', 'marginRight': '10px'}),
                            dcc.Input(
                                id={'type': 'boundary-input', 'index': i},
                                type="text",
                                value=auto_boundary,
                                style={
                                    'flex': '1', 'fontSize': '1em', 'margin': '5px', 'width': '80%',
                                    'backgroundColor': '#1e1e1e', 'color': '#DDDDDD',
                                    'border': '1px solid #555'
                                }
                            ),
                            html.Button("Запомнить", id={'type': 'submit-boundary', 'index': i}, style={
                                'backgroundColor': '#1e1e1e', 'color': '#DDDDDD',
                                'border': '1px solid #555'
                            }),
                        ]),
                    ]
                )
            )
        return inputs