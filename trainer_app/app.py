import dash
from layout.layout import create_layout
from callbacks.direct_task_callback import register_direct_task_callback
from callbacks.update_Pe_callback import register_Pe_callback
from callbacks.boundaries_callback import register_boundaries_callback
from callbacks.debits_callback import register_debits_callback
from callbacks.animation_callback import register_animation_callback
from callbacks.table_callback import register_table_callback
from callbacks.iterations_callback import register_iterations_callback
from callbacks.residuals_callback import register_residuals_callback
from callbacks.hist_callback import register_hist_callback
from callbacks.realisation_callback import register_realisation_callback
from callbacks.cache_callback import register_cache_callback

app = dash.Dash(__name__)

app.layout = create_layout()

register_direct_task_callback(app)
register_Pe_callback(app)
register_boundaries_callback(app)
register_debits_callback(app)
register_animation_callback(app)
register_table_callback(app)
register_iterations_callback(app)
register_residuals_callback(app)
register_hist_callback(app)
register_realisation_callback(app)
register_cache_callback(app)

if __name__ == '__main__':
    app.run_server(debug=True, port=8052)