import pandas as pd

def process_results(param_history):
    df = pd.DataFrame(param_history, columns=['parameters', 'Невязка', 'Итерация'])
    params_df = pd.json_normalize(df['parameters'])

    if 'Pe' in params_df:
        pe_list = params_df.pop('Pe')
        pe_df = pd.DataFrame(
            [pe[1:-1] for pe in pe_list],
            columns=[f"Pe_{i}" for i in range(1, len(pe_list[0]) - 1)]
        )
        params_df = pd.concat([params_df, pe_df], axis=1)

    return pd.concat([df.drop(columns='parameters'), params_df], axis=1)