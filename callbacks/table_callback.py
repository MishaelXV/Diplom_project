import pandas as pd
from dash.dependencies import Input, Output
from block.calculates import round_mantissa

def prepare_dataframe(df_history):
    df_history['Невязка'] = df_history['Невязка'].apply(round_mantissa)
    df_history.reset_index(inplace=True)
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history.to_dict('records')


def register_table_callback(app):
    @app.callback(
        Output('table-graph', 'data'),
        Input('optimization-cache', 'data'),
        prevent_initial_call=True
    )
    def update_table(optimization_data):
        try:
            # Проверяем наличие данных в кэше
            if not optimization_data or 'df_history' not in optimization_data:
                return [{'Итерация': 'Нет данных', 'Невязка': 'Дождитесь вычислений'}]

            # Восстанавливаем DataFrame из кэша
            df = pd.DataFrame(
                data=optimization_data['df_history']['data'],
                columns=optimization_data['df_history']['columns']
            )

            # Если prepare_dataframe возвращает DataFrame
            prepared_df = prepare_dataframe(df)

            # Преобразуем в формат для DataTable
            if isinstance(prepared_df, pd.DataFrame):
                return prepared_df.to_dict('records')
            elif isinstance(prepared_df, list):
                return prepared_df  # Если уже список словарей
            else:
                raise ValueError("Неподдерживаемый формат данных")

        except Exception as e:
            return [{
                'Итерация': 'Ошибка',
                'Невязка': str(e),
                'Параметры': 'Недоступны'
            }]