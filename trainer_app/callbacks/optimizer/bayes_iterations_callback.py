from dash.dependencies import Input, Output

def register_bayes_iterations_callback(app):
    @app.callback(
        Output('bayes-params-container', 'style'),
        Input('optimizer-method', 'value')
    )
    def show_bayes_params(method):
        if method == 'bayes':
            return {'display': 'block', 'marginBottom': '10px'}
        return {'display': 'none'}