import dash
from dash.dependencies import Input, Output
from callbacks.boundaries import extract_boundaries
from block.calculates import round_mantissa
from block.calculates import perform_optimization

def prepare_dataframe(df_history):
    df_history['Невязка'] = df_history['Невязка'].apply(round_mantissa)
    df_history.reset_index(inplace=True)
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history.to_dict('records')


def register_table_callback(app):
    @app.callback(
        Output('table-graph', 'data'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value')]
    )
    def update_table(b_values, boundary_values, A, TG0, atg, sigma, N):
        try:
            left_boundary, right_boundary = extract_boundaries(boundary_values)

            result, param_history, df_history, x_data, y_data = perform_optimization(
                left_boundary, right_boundary, b_values, TG0, atg, A, sigma, N)

            records = prepare_dataframe(df_history)

            return records

        except Exception as e:
            return [{'Итерация': 'Ошибка', 'Невязка': str(e)}]