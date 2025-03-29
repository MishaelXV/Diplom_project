import dash
from dash.dependencies import Input, Output
from block.calculates import perform_optimization
from components.boundaries import extract_boundaries

def register_cache_callback(app):
    @app.callback(
        Output('optimization-cache', 'data'),
        Output('boundaries-cache', 'data'),
        Input('solve_inverse_task', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')],
        prevent_initial_call=True
    )
    def compute_and_cache(n_clicks, b_values, boundary_data, A, TG0, atg, sigma, N):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        # 1. Извлекаем границы
        left, right = extract_boundaries(boundary_data)

        # 2. Вычисляем оптимизацию
        result, param_history, df_history, x_data, y_data = perform_optimization(
            left, right, b_values, TG0, atg, A, sigma, N
        )

        # 3. Подготавливаем данные для кэша (только сериализуемые объекты)
        boundaries_cache = {
            'left': left,
            'right': right
        }

        # Преобразуем param_history
        serializable_param_history = [
            {
                'params': dict(params),  # Преобразуем в обычный словарь
                'residual': float(residual)
            }
            for params, residual in param_history
        ]

        # Преобразуем DataFrame
        df_history_dict = {
            'columns': df_history.columns.tolist(),
            'data': df_history.values.tolist()
        }

        # Преобразуем numpy массивы
        optimization_cache = {
            'params': {k: v.value for k, v in result.params.items()},  # Основные параметры
            'success': bool(result.success),
            'message': str(result.message),
            'param_history': serializable_param_history,
            'df_history': df_history_dict,
            'x_data': x_data.tolist(),
            'y_data': y_data.tolist(),
            'chisqr': float(result.chisqr) if hasattr(result, 'chisqr') else None,
            'redchi': float(result.redchi) if hasattr(result, 'redchi') else None
        }

        return optimization_cache, boundaries_cache