import dash
from dash import html
from dash.dependencies import Input, Output, State
from main_block.main_functions import debit


def styled_message(text, color='#DDDDDD'):
    return html.Div(text, style={'color': color, 'fontWeight': 'normal', 'margin': '5px 0'})


def register_debits_callback(app):
    @app.callback(
        Output("debits-output", "children"),
        [
            Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
            Input("calculate-debits-btn", "n_clicks"),
            State('boundary-store', 'data'),
        ]
    )
    def update_debits(pe_values, n_clicks, boundary_values):
        if not n_clicks:
            return styled_message("Нажмите 'Вычислить расходы', чтобы обновить значения.")

        if not pe_values or any(pe is None or pe == '' for pe in pe_values):
            return styled_message("Расходы не вычислены. Убедитесь, что заданы значения Пекле для всех участков.")

        try:
            pe_values = [float(pe) for pe in pe_values]

            if len(pe_values) <= 1:
                return styled_message("Общее число участков должно быть больше одного. Проверьте введённые данные.")

            if any(pe < 0 for pe in pe_values):
                return styled_message("Ошибка: значения Пекле не могут быть отрицательными. Проверьте введённые данные.")

            if not boundary_values:
                return styled_message("Ошибка: Границы интервалов отсутствуют.")

            left_boundary = boundary_values.get('left', [])
            right_boundary = boundary_values.get('right', [])

            if len(left_boundary) != len(pe_values) or len(right_boundary) != len(pe_values):
                return styled_message("Ошибка: Количество границ не соответствует числу значений Пекле.")

            if all(value == 0 for value in left_boundary) or all(value == 0 for value in right_boundary):
                return styled_message("Ошибка: Границы интервалов отсутствуют.")

            debits = [debit(pe) for pe in pe_values]

            debits_elements = [
                html.Div(
                    f"Изолированный участок {i + 1}: Расход = {debit_value:.5f} м³/сут",
                    style={'color': '#DDDDDD', 'margin': '4px 0'}
                )
                for i, debit_value in enumerate(debits[:-1])
            ]

            return debits_elements

        except ValueError:
            return styled_message("Ошибка: все значения Пекле должны быть числовыми.")
        except Exception as e:
            return styled_message(f"Ошибка при вычислении дебитов: {e}")