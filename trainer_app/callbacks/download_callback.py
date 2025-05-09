from dash import Input, Output, State, dcc
import plotly.io as pio
from PyPDF2 import PdfMerger
import os
import plotly.graph_objects as go

def register_download_callback(app):
    @app.callback(
        Output("download-pdf", "data"),
        Input("export_to_pdf", "n_clicks"),
        State("stored-figures", "data"),
        prevent_initial_call=True,
    )
    def generate_pdf(n_clicks, stored_figures):
        if not stored_figures:
            return None

        if isinstance(stored_figures, dict):
            stored_figures = [stored_figures]

        merger = PdfMerger()
        temp_files = []

        for i, fig_json in enumerate(stored_figures):
            fig = go.Figure(fig_json)
            filename = f"temp_fig_{i}.pdf"
            pio.write_image(fig, filename)
            merger.append(filename)
            temp_files.append(filename)

        output_filename = "final_report.pdf"
        merger.write(output_filename)
        merger.close()

        # Читаем содержимое PDF файла
        with open(output_filename, "rb") as f:
            pdf_content = f.read()

        # Удаляем временные файлы
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # Возвращаем файл напрямую (Dash автоматически обработает скачивание)
        return dcc.send_bytes(pdf_content, "отчет.pdf")