from dash.dependencies import Input, Output, State

def register_optimizer_callback(app):
    @app.callback(
        Output('optimizer-method-container', 'style'),
        Input('open-optimizers', 'n_clicks'),
        prevent_initial_call=True
    )
    def show_optimizer_method_dropdown(n_clicks):
        if n_clicks:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}