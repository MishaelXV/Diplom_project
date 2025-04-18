import time
from dash.dependencies import Input, Output

def register_realisation_callback(app):
    @app.callback(
        Output('detal-container', 'style'),
        Input('solve_inverse_task', 'n_clicks'),
    )
    def update_realisation_container(n_clicks):
        if not n_clicks:
            return {'display': 'none'}
        else:
            time.sleep(4),
            return {
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'flex-start',
                'width': '100%'
            }