from dash import Output, Input, dcc, dash

def register_loading_callback(app):
    @app.callback(
        Output('animation-loading', 'children'),
        Output('animation-container', 'style'),
        Output('parallel-loading', 'children'),
        Output('parallel-coordinates-graph', 'style'),
        Input('optimization-cache', 'data'),
        Input('boundaries-cache', 'data'),
        prevent_initial_call=True
    )
    def update_loading_indicators(optimization_data, boundaries_data):
        if not optimization_data or not boundaries_data:
            return (
                dcc.Loading(type="circle", color="#00cc96"),
                {'display': 'none'},
                dcc.Loading(type="circle", color="#00cc96"),
                {'display': 'none'}
            )
        else:
            return (
                dash.no_update,
                {
                    'display': 'block',
                    'width': '98.5%',
                    'height': '450px',
                    'border': '1px solid #444444',
                    'padding': '10px',
                    'boxSizing': 'border-box',
                    'backgroundColor': '#000000'
                },
                dash.no_update,
                {
                    'width': '100%',
                    'height': '400px',
                    'display': 'block'
                }
            )