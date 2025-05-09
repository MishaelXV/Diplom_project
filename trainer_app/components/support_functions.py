import re
from main_block.main_functions import main_func

def extract_boundaries(boundary_values):
    left_boundary = boundary_values.get('left', [])
    right_boundary = boundary_values.get('right', [])
    left_boundary = [float(b) for b in left_boundary]
    right_boundary = [float(b) for b in right_boundary]
    return left_boundary, right_boundary


def round_mantissa(value, decimals=5):
    formatted = f"{value:.{decimals}e}"
    return float(formatted)


def prepare_dataframe(df_history):
    df_history = df_history.loc[:, ~df_history.columns.str.startswith('delta_')]
    df_history['Невязка'] = df_history['Невязка'].apply(round_mantissa)
    df_history.reset_index(inplace=True)
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    df_history.rename(columns={'Невязка': 'J', 'residuals': 'E'}, inplace=True)

    new_columns = {}
    for col in df_history.columns:
        match = re.match(r'^Pe_(\d+)$', col)
        if match:
            idx = int(match.group(1))
            new_columns[col] = f'Pe_{idx + 1}'

    df_history.rename(columns=new_columns, inplace=True)

    return df_history.to_dict('records')


def prepare_dataframe_2(df_history):
    df_history = df_history.reset_index()
    df_history.rename(columns={'index': 'Итерация'}, inplace=True)
    return df_history


def residuals(x, y, TG0, atg, A, Pe_opt, left_boundaries, right_boundaries):
    return main_func(x, TG0, atg, A, Pe_opt, left_boundaries, right_boundaries) - y