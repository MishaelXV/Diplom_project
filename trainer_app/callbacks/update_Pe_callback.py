from dash import dcc, html
from dash.dependencies import Input, Output

def register_Pe_callback(app):
    @app.callback(
        Output('dynamic-b-inputs', 'children'),
        Input('a-input', 'value')
    )
    def update_b_inputs(num_b):
        if num_b == 1:
            b_values_initial = [2000]
        else:
            b_values_initial = [
                round(2000 - (2000 / (num_b - 1)) * i)
                for i in range(num_b)
            ]

        def generate_boundaries(index):
            base = index * 150
            return f"{base} {base + 100}"

        inputs = []
        for i in range(num_b):
            auto_boundary = generate_boundaries(i)

            inputs.append(
                html.Div(
                    style={'marginBottom': '10px', 'position': 'relative'},
                    children=[
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center'},
                            children=[
                                html.Label(f"{i + 1} изолированный интервал:",
                                           style={'color': '#DDDDDD', 'marginRight': '10px'}),
                            ]
                        ),
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center', 'position': 'relative'},
                            children=[
                                html.Label("Pe:",
                                           id={'type': 'label-b-input', 'index': i},
                                           style={'color': '#DDDDDD', 'marginRight': '10px', 'cursor': 'pointer'}),
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
                                html.Div(
                                    id={'type': 'info-b-input', 'index': i},
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
                                )
                            ]
                        ),
                        html.Div(
                            style={'display': 'flex', 'alignItems': 'center'},
                            children=[
                                html.Label("Границы интервала:", style={'color': '#DDDDDD', 'marginRight': '10px'}),
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
                            ]
                        ),
                    ]
                )
            )
        return inputs