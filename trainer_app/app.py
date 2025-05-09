import dash
from layout.layout import create_layout
from callbacks.direct_task_callback import register_direct_task_callback
from callbacks.update_Pe_callback import register_Pe_callback
from callbacks.boundaries_callback import register_boundaries_callback
from callbacks.debits_callback import register_debits_callback
from trainer_app.callbacks.animation.animation_callback import register_animation_callback
from callbacks.table_callback import register_table_callback
from callbacks.iterations_callback import register_iterations_callback
from callbacks.residuals_callback import register_residuals_callback
from trainer_app.callbacks.optimizer.cache_callback import register_cache_callback
from callbacks.params_info_callback import register_params_info_callback
from trainer_app.callbacks.animation.load_callback import register_load_callback
from trainer_app.callbacks.details_container_visible_callback import register_details_container_visible_callback
from trainer_app.callbacks.download_callback import register_download_callback
from trainer_app.callbacks.loading_callback import register_loading_callback
from trainer_app.callbacks.metrics_callback import register_metrics_callback
from trainer_app.callbacks.optimizer.optimizer_callback import register_optimizer_callback
from trainer_app.callbacks.parallel_graph_callback import register_parallel_graph_callback

app = dash.Dash(__name__)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                background-color: #111111;
                margin: 0;
                padding: 0;
                overflow-x: hidden;
                color: white;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = create_layout()
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/brPBPO.css'
})

app.css.append_css({
    'external_url': '''
        .modal-content {
            background-color: #1e1e1e !important;
            color: #DDDDDD !important;
            border: 1px solid #444 !important;
        }
        .modal-header, .modal-footer {
            border-color: #444 !important;
        }
        .close {
            color: #DDDDDD !important;
        }
    '''
})

register_direct_task_callback(app)
register_Pe_callback(app)
register_boundaries_callback(app)
register_debits_callback(app)
register_animation_callback(app)
register_table_callback(app)
register_iterations_callback(app)
register_residuals_callback(app)
register_cache_callback(app)
register_params_info_callback(app)
register_load_callback(app)
register_details_container_visible_callback(app)
register_parallel_graph_callback(app)
register_download_callback(app)
register_loading_callback(app)
register_metrics_callback(app)
register_optimizer_callback(app)

if __name__ == '__main__':
    app.run(debug=True, port=8052)