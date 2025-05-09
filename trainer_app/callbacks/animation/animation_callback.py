import dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from trainer_app.components.graphs import create_figure_animation, generate_frames

def register_animation_callback(app):
    @app.callback(
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
            return dash.no_update

        left_boundary = boundaries_data['left']
        right_boundary = boundaries_data['right']

        fixed_first_pe = optimization_data['fixed_pe']['first']
        fixed_last_pe = optimization_data['fixed_pe']['last']

        df_history = pd.DataFrame(
            data=optimization_data['df_history']['data'],
            columns=optimization_data['df_history']['columns']
        )

        pe_columns = [col for col in df_history.columns if col.startswith('Pe_')]

        param_history = [
            (df_history.loc[i, pe_columns].tolist(), df_history.loc[i, 'Невязка'])
            for i in range(len(df_history))
        ]
        print(param_history)
        x_data = np.array(optimization_data['x_data'])
        x_data_true = boundaries_data['x_data_true']
        y_data_true = boundaries_data['y_data_true']

        frames = generate_frames(
            param_history=param_history,
            x_data=x_data,
            x_data_true=x_data_true,
            y_data_noize=y_data_true,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            TG0=TG0,
            atg=atg,
            A=A,
            fixed_first_pe=fixed_first_pe,
            fixed_last_pe=fixed_last_pe
        )

        fig = create_figure_animation(
            frames=frames,
            x_data=x_data,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            TG0=TG0,
            atg=atg,
            A=A,
            b_values=b_values,
            x_data_true= x_data_true,
            y_data_noize = y_data_true,
        )

        return fig