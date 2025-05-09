from dash import Output, Input, html

def register_metrics_callback(app):
    @app.callback(
        Output('optimization-metrics-content', 'children'),
        Input('optimization-cache', 'data'),
    )
    def update_optimization_metrics(opt_cache):
        if not opt_cache or 'metrics' not in opt_cache:
            return html.Div("Метрики недоступны")

        metrics = opt_cache['metrics']

        readable_names = {
            "Метод оптимизации": "Метод оптимизации",
            "MAE": "Средняя абсолютная ошибка (MAE)",
            "MSE": "Среднеквадратичная ошибка (MSE)",
            "RMSE": "Квадратный корень из MSE (RMSE)",
            "R2": "Коэффициент детерминации (R²)",
            "MAPE": "Относительная ошибка (MAPE) [%]",
            "Chi2": "Хи-квадрат (χ²)",
            "Runtime_sec": "Время выполнения [сек]"
        }

        return html.Div(
            style={
                'display': 'flex',
                'flexDirection': 'column',
                'gap': '8px',
                'overflowX': 'hidden',
                'fontFamily': 'Arial, sans-serif',
                'padding': '10px',
                'border': '1px solid #444444',
                'borderRadius': '8px',
                'backgroundColor': '#000000',
            },
            children=[
                html.Div(
                    [
                        html.Span(
                            f"{readable_names.get(key, key)}: ",
                            style={
                                'fontWeight': 'bold',
                                'display': 'inline-block',
                                'width': '300px',
                                'color': 'white'
                            }
                        ),
                        html.Span(
                            f"{value:.5f}" if isinstance(value, (float, int)) else str(value),
                            style={
                                'display': 'inline-block',
                                'color': 'white',
                                'fontFamily': 'monospace'
                            }
                        )
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'center',
                    }
                )
                for key, value in metrics.items()
            ]
        )