import plotly.io as pio
from dash.dependencies import Input, Output, State

pio.templates.default = "seaborn"

def register_callbacks(app):
    @app.callback(
        [Output('animation-graph', 'style')],
        [Input('fullscreen-button', 'n_clicks')],
        [State('animation-graph', 'style')]
    )
    def toggle_fullscreen(n_clicks, graph_style):
        if n_clicks % 2 == 1:
            return [{
                'height': '100vh',
                'width': '100vw',
                'position': 'fixed',
                'top': '0',
                'left': '0',
                'zIndex': 1000
            }, ]
        else:
            return [{
                'height': '400px',
                'width': '100%',
                'position': 'relative',
            }]








