from dash import callback_context, dcc, ALL
from dash.dependencies import Input, Output, State


def register_params_info_callback(app):
    static_param_ids = ["a-input", "A-input", "TG0-input", "atg-input", "sigma-input", "N-input"]

    app.callback(
        [Output(f"info-{param}", "style") for param in static_param_ids] +
        [Output(f"info-{param}", "children") for param in static_param_ids],
        [Input(f"label-{param}", "n_clicks") for param in static_param_ids],
    )(lambda *clicks: update_static_info_panels(clicks, static_param_ids))

    app.callback(
        [Output({'type': 'info-b-input', 'index': ALL}, "style"),
         Output({'type': 'info-b-input', 'index': ALL}, "children")],
        [Input({'type': 'label-b-input', 'index': ALL}, "n_clicks")],
        [State({'type': 'label-b-input', 'index': ALL}, "id")]
    )(update_dynamic_info_panels)


def update_static_info_panels(clicks, param_ids):
    ctx = callback_context
    if not ctx.triggered:
        styles = [{'display': 'none'} for _ in param_ids]
        texts = [None for _ in param_ids]
        return styles + texts

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]

    param_info = {
        "a-input": "Количество изолированных участков скважины — определяет количество сегментов модели.",
        "A-input": dcc.Markdown(r"""
            Параметр теплоотдачи A:

            $$\text{A} = \frac{\lambda_w}{\lambda_s}\frac{1}{\ln(r_s/r_w)}$$
            
            Где:
            - $\lambda_s$ — теплопроводность пласта `[Вт/(м·K)]`
            - $\lambda_w$ — теплопроводность воды `[Вт/(м·K)]`
            - $r_w$ — радиус скважины `[м]`
            - $r_s$ — радиус пласта `[м]`
            
            """, mathjax=True),
                    "TG0-input": "Параметр TG0 - температура на поверхности пласта.",
                    "atg-input": "Параметр tg - геотермальный градиент (скорость роста температуры с глубиной)",
                    "sigma-input": "Параметр sigma - стандартное отклонение уровня шума в данных.",
                    "N-input": "Параметр N - количество точек замера по глубине скважины.",
    }

    styles = []
    texts = []

    for i, param_id in enumerate(param_ids):
        click_count = clicks[i] or 0

        if f"label-{param_id}" == clicked_id:
            if click_count % 2 == 1:
                styles.append({
                    'display': 'block',
                    'color': '#DDDDDD',
                    'fontSize': '14px',
                    'padding': '5px',
                    'backgroundColor': '#333333',
                    'borderRadius': '5px',
                    'marginTop': '5px'
                })
                texts.append(param_info.get(param_id, None))
            else:
                styles.append({'display': 'none'})
                texts.append(None)
        else:
            if click_count % 2 == 1:
                styles.append({
                    'display': 'block',
                    'color': '#DDDDDD',
                    'fontSize': '14px',
                    'padding': '5px',
                    'backgroundColor': '#333333',
                    'borderRadius': '5px',
                    'marginTop': '5px'
                })
                texts.append(param_info.get(param_id, None))
            else:
                styles.append({'display': 'none'})
                texts.append(None)

    return styles + texts


def update_dynamic_info_panels(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return [{'display': 'none'} for _ in ids], [None for _ in ids]

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    trigger_idx = eval(trigger_id)['index']

    pe_info = dcc.Markdown(r"""
    **Параметр Pe (число Пекле) для интервала {i}**:

    $$\text{{Pe}} = \frac{{\rho_w c_w q L}}{{\lambda_s}}$$

    Где:
    - $\rho_w$ — плотность воды `[кг/м³]`
    - $c_w$ — удельная теплоемкость воды `[Дж/(кг·K)]`
    - $q$ — объемный расход воды `[м³/с]`
    - $L$ — характерная длина (длина интервала) `[м]`
    - $\lambda_s$ — теплопроводность пласта `[Вт/(м·K)]`

    """.format(i=trigger_idx + 1), mathjax=True)

    styles = []
    texts = []

    for i, component_id in enumerate(ids):
        idx = component_id['index']
        click_count = clicks[i] or 0

        if idx == trigger_idx:
            if click_count % 2 == 1:
                styles.append({
                    'display': 'block',
                    'color': '#DDDDDD',
                    'fontSize': '14px',
                    'padding': '5px',
                    'backgroundColor': '#333333',
                    'borderRadius': '5px',
                    'marginTop': '5px'
                })
                texts.append(pe_info)
            else:
                styles.append({'display': 'none'})
                texts.append(None)
        else:
            if click_count % 2 == 1:
                styles.append({
                    'display': 'block',
                    'color': '#DDDDDD',
                    'fontSize': '14px',
                    'padding': '5px',
                    'backgroundColor': '#333333',
                    'borderRadius': '5px',
                    'marginTop': '5px'
                })
                texts.append(pe_info)
            else:
                styles.append({'display': 'none'})
                texts.append(None)

    return styles, texts