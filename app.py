import dash
from layout.layout import create_layout
from callbacks.callbacks import register_callbacks
from callbacks.direct_task_callback import register_direct_task_callback

app = dash.Dash(__name__)

app.layout = create_layout()
register_callbacks(app)
register_direct_task_callback(app)

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)