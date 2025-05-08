import pandas as pd
from dash.dependencies import Input, Output
from trainer_app.components.support_functions import prepare_dataframe

def register_table_callback(app):
    @app.callback(
        Output('table-graph', 'data'),
        Input('optimization-cache', 'data'),
        prevent_initial_call=True
    )
    def update_table(optimization_data):
        try:
            if not optimization_data or 'df_history' not in optimization_data:
                return [{'Итерация': 'Нет данных', 'Невязка': 'Дождитесь вычислений'}]

            df = pd.DataFrame(
                data=optimization_data['df_history']['data'],
                columns=optimization_data['df_history']['columns']
            )

            prepared_df = prepare_dataframe(df)

            if isinstance(prepared_df, pd.DataFrame):
                return prepared_df.to_dict('records')
            elif isinstance(prepared_df, list):
                return prepared_df
            else:
                raise ValueError("Неподдерживаемый формат данных")

        except Exception as e:
            return [{
                'Итерация': 'Ошибка',
                'Невязка': str(e),
                'Параметры': 'Недоступны'
            }]