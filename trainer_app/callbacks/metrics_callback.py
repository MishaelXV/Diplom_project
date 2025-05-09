from dash import Output, Input, html

def register_metrics_callback(app):
    @app.callback(
        Output('optimization-metrics-content', 'children'),
        Input('optimization-cache', 'data'),
    )
    def update_optimization_metrics(input_value):
        metrics = {
            "Целевая функция": 0.123,
            "Итерации": 150,
            "Время выполнения": "45.2 сек",
            "Сходимость": "Достигнута",
            "Точность": 1e-4,
        }

        return html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '8px',
                'overflowX': 'hidden'
            },
            children=[
                html.Div(
                    [
                        html.Span(
                            f"{name}: ",
                            style={
                                'fontWeight': 'bold',
                                'display': 'inline-block',
                                'minWidth': '120px'
                            }
                        ),
                        html.Span(
                            str(value),
                            style={
                                'display': 'inline-block',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'whiteSpace': 'nowrap'
                            }
                        )
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'width': '100%'
                    }
                )
                for name, value in metrics.items()
            ]
        )