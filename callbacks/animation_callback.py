import dash
import numpy as np
from dash.dependencies import Input, Output
from components.graphs import create_figure_animation, generate_frames

def register_animation_callback(app):
    @app.callback(
        Output('animation-container', 'style'),
        Output('animation-graph', 'figure'),
        Input('optimization-cache', 'data'),
        Input('boundaries-cache', 'data'),
        [Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value')],
        prevent_initial_call=True
    )
    def update_animation(optimization_data, boundaries_data, A, TG0, atg, b_values):
        if not optimization_data or not boundaries_data:
            return {'display': 'none'}, dash.no_update

        # 1. Восстанавливаем границы
        left_boundary = boundaries_data['left']
        right_boundary = boundaries_data['right']

        # 2. Восстанавливаем param_history
        param_history = [
            (item['params'], item['residual'])
            for item in optimization_data['param_history']
        ]

        # 3. Восстанавливаем массивы данных
        x_data = np.array(optimization_data['x_data'])
        y_data = np.array(optimization_data['y_data'])


        # 5. Генерация анимации с восстановленными данными
        frames = generate_frames(
            param_history=param_history,
            x_data=x_data,
            y_data=y_data,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            TG0=TG0,
            atg=atg,
            A=A,
            b_values=b_values
        )

        fig = create_figure_animation(
            frames=frames,
            x_data=x_data,
            param_history=param_history,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            TG0=TG0,
            atg=atg,
            A=A,
            b_values=b_values,
            y_data=y_data
        )

        return {
            'display': 'block',
            'width': '98.5%',
            'height': '450px',
            'border': '1px solid #ccc',
            'padding': '10px',
            'boxSizing': 'border-box',
            'backgroundColor': '#ffffff'
        }, fig