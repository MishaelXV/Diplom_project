from dash import Output, Input

def register_details_container_visible_callback(app):
    @app.callback(
        Output('detal-container', 'style'),
        Input('animation-loaded', 'data')
    )
    def toggle_details_container(loaded):
        if loaded:
            return {
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'flex-start',
                'width': '100%'
            }
        return {'display': 'none'}