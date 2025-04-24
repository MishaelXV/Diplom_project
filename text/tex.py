from pylatex import Document, Math, Section, NoEscape

doc = Document()

with doc.create(Section('Метрики качества для границ сегментов')):
    # MAE
    doc.append(Math(data=NoEscape(
        r"\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \left| \hat{b}_i - b_i \right|"
    )))

    # MSE
    doc.append(Math(data=NoEscape(
        r"\text{MSE} = \sum_{i=1}^{N} \left( (\hat{l}_i + p_i)^2 + (\hat{r}_i + p_i)^2 \right)"
    )))

    # RMSE
    doc.append(Math(data=NoEscape(
        r"\text{RMSE} = \sqrt{\text{MSE}}"
    )))

    # Relative MAE
    doc.append(Math(data=NoEscape(
        r"\text{Relative MAE} = \frac{\text{MAE}}{\sum_{i=1}^{N} (r_i - l_i)} \cdot 100\%"
    )))

doc.generate_pdf("boundary_metrics", clean_tex=False)