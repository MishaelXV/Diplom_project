import dash
import time
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from main_block.main_functions import main_func
from dash.dependencies import Input, Output, State
from optimizator.bayes_optimizer import run_bayes_optimization
from optimizator.optimizer import run_optimization
from regression.global_models import model_ws, model_ms
from trainer_app.components.support_functions import extract_boundaries
from regression.find_intervals import get_boundaries
from main_block.data import generate_data, noize_data

def register_cache_callback(app):
    @app.callback(
        Output('optimization-cache', 'data'),
        Output('boundaries-cache', 'data'),
        Input('run-optimization-btn', 'n_clicks'),
        [Input({'type': 'b-input', 'index': dash.dependencies.ALL}, 'value'),
         Input('boundary-store', 'data'),
         Input('A-input', 'value'),
         Input('TG0-input', 'value'),
         Input('atg-input', 'value'),
         Input('sigma-input', 'value'),
         Input('N-input', 'value'),
         Input('bayes-iterations', 'value')],  # Добавляем ввод количества итераций
        State('optimizer-method', 'value'),
        prevent_initial_call=True
    )
    def compute_and_cache(n_clicks, b_values, boundary_data, A, TG0, atg, sigma, N, n_trials, optimizer_method):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        start_time = time.time()

        left_true, right_true = extract_boundaries(boundary_data)
        x_data, y_data = generate_data(left_true, right_true, b_values, TG0, atg, A, N)
        y_data_noize = noize_data(y_data, sigma)

        found_left, found_right = get_boundaries(
            x_data, y_data_noize, b_values, N, sigma, A, model_ws, model_ms
        )

        if optimizer_method == 'bayes':
            trials = n_trials if n_trials is not None else 200
            Pe_opt, df_history = run_bayes_optimization(x_data, y_data_noize, found_left, found_right,
                           left_true, right_true, b_values, TG0, atg, A, trials)

        else:
            Pe_opt, df_history = run_optimization(
                x_data, y_data_noize, found_left, found_right,
                boundary_data['left'], boundary_data['right'],
                b_values, TG0, atg, A, optimizer_method
            )

        y_pred = main_func(x_data, TG0, atg, A, Pe_opt, found_left, found_right)

        mae = mean_absolute_error(y_data_noize, y_pred)
        mse = mean_squared_error(y_data_noize, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_data_noize, y_pred)
        mape = np.mean(np.abs((y_data_noize - y_pred) / y_data_noize)) * 100
        eps = 1e-8
        chi2 = np.sum(((y_data_noize - y_pred) ** 2) / (y_pred + eps))
        duration = time.time() - start_time

        boundaries_cache = {
            'left': found_left,
            'right': found_right,
            'left_true': left_true,
            'right_true': right_true,
            'x_data_true': x_data,
            'y_data_true': y_data_noize,
        }

        df_history_dict = {
            'columns': df_history.columns.tolist(),
            'data': df_history.values.tolist()
        }

        optimization_cache = {
            'found_Pe': Pe_opt,
            'true_Pe': b_values,
            'df_history': df_history_dict,
            'x_data': x_data.tolist(),
            'y_data': y_data.tolist(),
            'fixed_pe': {
                'first': b_values[0],
                'last': b_values[-1]
            },
            'metrics': {
                'Метод оптимизации': optimizer_method,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Chi2': chi2,
                'Runtime_sec': duration,
                'Количество итераций (Bayes)': n_trials if optimizer_method == 'bayes' else None
            }
        }

        return optimization_cache, boundaries_cache