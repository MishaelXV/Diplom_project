import dash
from dash.dependencies import Input, Output, State

def register_boundaries_callback(app):
    @app.callback(
        Output('boundary-store', 'data'),
        Input({'type': 'submit-boundary', 'index': dash.dependencies.ALL}, 'n_clicks'),
        [State({'type': 'boundary-input', 'index': dash.dependencies.ALL}, 'value')]
    )
    def save_boundaries(n_clicks, boundary_values):
        if n_clicks and any(n_clicks):
            left_boundary = []
            right_boundary = []
            for i, n in enumerate(n_clicks):
                boundaries = boundary_values[i].strip().split()
                if len(boundaries) >= 2:
                    left_boundary.append(float(boundaries[0]))
                    right_boundary.append(float(boundaries[1]))
                else:
                    print(f"Некорректные границы для интервала {i + 1}: {boundary_values[i]}")

            return {'left': left_boundary, 'right': right_boundary}