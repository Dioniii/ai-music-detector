"""Streamlit portfolio showcase for the saved audio-feature baseline.

Run with: python -m streamlit run app.py
"""
import base64
import html
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import streamlit as st

from showcase import (
    ROOT, DEFAULT_OUTPUT, FONT_DIR, EMPTY, analyze, confusion_figure,
    distribution_figure, local_examples, read_csv,
)
from showcase_content import (
    HERO, RESULTS_INTRO, SCORES_INTRO, LIMITATIONS, HOW_IT_WORKS,
    CONTRIBUTION_GUIDE, INPUT_GUIDANCE,
)


@st.cache_data(show_spinner=False)
def stylesheet():
    """Bundle local fonts; the browser makes no third-party font requests."""
    fonts = []
    for name, weight in [('Regular', 400), ('SemiBold', 600), ('Bold', 700)]:
        encoded = base64.b64encode((FONT_DIR / f'IBMPlexMono-{name}.ttf').read_bytes()).decode('ascii')
        fonts.append(f"@font-face {{font-family:'IBM Plex Mono';src:url(data:font/ttf;base64,{encoded}) format('truetype');font-weight:{weight};font-display:swap;}}")
    return '<style>' + ''.join(fonts) + (ROOT / 'assets/streamlit.css').read_text(encoding='utf-8') + '</style>'


@st.cache_resource
def analysis_lock():
    # Matplotlib is shared by sessions; serialize plotting and bound decode concurrency.
    return RLock()


def figure_png(figure):
    output = io.BytesIO()
    figure.savefig(output, format='png', dpi=130, facecolor=figure.get_facecolor())
    figure.clear()
    return output.getvalue()


def run_analysis(data, name):
    """Decode an isolated temporary file and delete it even when analysis fails.

    Only this browser session retains the result and original playback bytes.
    Uploaded audio is never placed in Streamlit's shared cache.
    """
    if len(data) > 50 * 1024 * 1024:
        return ('<div class="result-card"><h2>Could not analyze this recording</h2>'
                '<p>Please choose a recording smaller than 50 MB.</p></div>', '', None, None, None, [])
    with analysis_lock(), TemporaryDirectory(prefix='music-detector-') as directory:
        path = Path(directory) / ('recording' + Path(name).suffix.lower())
        path.write_bytes(data)
        verdict, details, original, signal, influence, rows = analyze(path, display_name=name)
        return (verdict, details, data if original else None,
                figure_png(signal) if signal is not None else None,
                figure_png(influence) if influence is not None else None, rows)


def clear_result():
    st.session_state.pop('analysis_result', None)
    st.session_state.pop('playback_format', None)


def upload_changed():
    clear_result()
    st.session_state['example'] = None


def example_changed():
    clear_result()
    # A new uploader key clears the previously uploaded file unambiguously.
    st.session_state['upload_generation'] = st.session_state.get('upload_generation', 0) + 1


def numeric_table(headers, rows):
    """Accessible HTML table keeps numeric values right-aligned and monospace."""
    head = ''.join(f'<th scope="col">{html.escape(value)}</th>' for value in headers)
    body = []
    for row in rows:
        cells = []
        for value in row:
            numeric = isinstance(value, (float, int))
            rendered = f'{value:.6g}' if numeric else str(value)
            cells.append(f'<td class="{"number" if numeric else "text"}">{html.escape(rendered)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    st.html('<div class="table-scroll"><table class="values-table"><thead><tr>' + head +
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


@st.cache_data(show_spinner=False)
def evaluation_artifacts():
    metrics = json.loads((DEFAULT_OUTPUT / 'metrics.json').read_text(encoding='utf-8'))['metrics']
    predictions = read_csv(DEFAULT_OUTPUT / 'predictions.csv')
    with analysis_lock():
        confusion = figure_png(confusion_figure(metrics))
        distribution = figure_png(distribution_figure(predictions))
    return metrics, predictions, confusion, distribution


def render_analysis():
    left, right = st.columns([1, 1.65], gap='large')
    with left:
        upload = st.file_uploader('Your recording', type=['wav', 'mp3', 'flac', 'ogg'],
                                  key=f'upload_{st.session_state.get("upload_generation", 0)}',
                                  on_change=upload_changed, max_upload_size=50)
        examples, labels = local_examples()
        example = None
        if examples:
            example = st.selectbox('Try an existing recording, then press Analyze',
                                   options=range(len(examples)), index=None,
                                   format_func=lambda index: labels[index], key='example',
                                   on_change=example_changed)
        st.markdown(INPUT_GUIDANCE)
        if st.button('Analyze recording', type='primary', width='stretch'):
            clear_result()
            if upload is not None or example is not None:
                with st.spinner('Analyzing recording...'):
                    if upload is not None:
                        data, name = upload.getvalue(), upload.name
                    else:
                        path = Path(examples[example][0])
                        data, name = path.read_bytes(), path.name
                    st.session_state['analysis_result'] = run_analysis(data, name)
                    st.session_state['playback_format'] = {'.mp3': 'audio/mpeg', '.flac': 'audio/flac', '.ogg': 'audio/ogg'}.get(Path(name).suffix.lower(), 'audio/wav')
            else:
                st.session_state['analysis_result'] = analyze(None)
        result = st.session_state.get('analysis_result', (EMPTY, '', None, None, None, []))
        if result[2] is not None:
            st.divider()
            st.markdown('Listen to the original')
            st.audio(result[2], format=st.session_state.get('playback_format', 'audio/wav'))
            st.markdown(result[1])
    with right:
        st.html(result[0])
        if result[3] is not None:
            st.image(result[3], caption='Inside the audio', width='stretch')
    st.divider()
    with st.expander('Why did the model lean this way?', expanded=True):
        st.markdown(CONTRIBUTION_GUIDE)
        if result[4] is not None:
            st.image(result[4], caption='Feature contributions', width='stretch')
        with st.expander('See the ten feature values'):
            numeric_table(['Feature', 'Measured value', 'Training-standardized value', 'Logit contribution'], result[5])


def render_results():
    metrics, predictions, confusion, distribution = evaluation_artifacts()
    h = metrics['holdout']
    cm = h['confusion_matrix_true_rows_predicted_columns_human_ai']
    st.markdown(RESULTS_INTRO)
    st.html(f'<div class="stat-grid"><div class="stat"><strong>{cm[1][1]} / {h["ai_tracks"]}</strong><span>AI recordings detected · recall {h["ai_recall"]:.1%}</span></div><div class="stat"><strong>{cm[0][1]} / {h["human_tracks"]}</strong><span>Human recordings falsely flagged · {h["human_false_positive_rate"]:.0%}</span></div><div class="stat"><strong>{h["ai_precision"]:.1%}</strong><span>AI precision on this small holdout</span></div></div>')
    st.image(confusion, caption='Correct predictions and errors', width='stretch')
    st.markdown(SCORES_INTRO)
    st.image(distribution, caption='Score distributions by dataset label', width='stretch')
    errors = [[r['reference'], r['generator'] or 'Human reference', r['true_label'], r['predicted_label'], float(r['ai_score'])]
              for r in predictions if r['split'] == 'holdout' and r['true_label'] != r['predicted_label']]
    st.markdown('The actual holdout mistakes')
    numeric_table(['Holdout recording', 'Source label', 'Dataset label', 'Prediction', 'AI score'], errors)
    st.markdown(LIMITATIONS)


def main():
    st.set_page_config(page_title='AI Music Detector · Audio Lab', layout='wide', initial_sidebar_state='collapsed')
    st.html(stylesheet())
    st.html(HERO)
    analysis_tab, results_tab, how_tab = st.tabs(['Analyze audio', 'Model results', 'How it works'])
    with analysis_tab:
        render_analysis()
    with results_tab:
        render_results()
    with how_tab:
        st.markdown(HOW_IT_WORKS)


if __name__ == '__main__':
    main()
