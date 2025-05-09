from dash.dependencies import Input, Output

def register_load_callback(app):
    @app.callback(
        Output('animation-loaded', 'data'),
        Input('animation-graph', 'figure')
    )
    def set_animation_loaded(fig):
        return fig is not None