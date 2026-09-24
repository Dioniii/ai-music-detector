"""Streamlit portfolio showcase for EfficientAT and our trained classifier.

Run with: python -m streamlit run app.py
"""
import base64
import html
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

import streamlit as st

from showcase import (
    ROOT, DEFAULT_OUTPUT, FONT_DIR, MODEL_PATH, EMPTY, analyze, confusion_figure,
    distribution_figure, local_examples, read_csv,
)

HERO = '<div class="hero"><h1>Human-made or AI-generated?</h1><p>Record nearby music or upload a song and see what the model thinks.</p><span class="small-note">An experimental music detector. It can make mistakes.</span></div>'

RESULTS_INTRO = """## How often did it get it right?
We used **700 recordings to teach the model**, **150 to help choose the best version**, and **150 more to check its predictions**. Each group had equal numbers of human and AI recordings. Related recordings stayed together to make the checks fairer.

The results below come from those checks, not from your upload. We have checked these recordings before while building the project, so this is not a completely new test.
"""

SCORES_INTRO = '### How the model scored each recording\nEach dot is one recording. Scores to the right of the middle line are labeled AI; scores to the left are labeled human. A human recording on the right, or an AI recording on the left, is a mistake.'

LIMITATIONS = '**Keep in mind:** the model can mistake human music for AI, and miss AI music too. These results cover Rock and Electronic music, not every genre. We have not checked how well it works with microphone recordings, background noise or new AI music tools. The labels come from the datasets; they are not proof of who made a song.'

HOW_IT_WORKS = """## What happens when you add a recording?
### 1. Pick a few parts to check
The app selects up to **five 20-second sections** from across your recording. This helps it look beyond just the intro. Sections can overlap. For recordings under 20 seconds, it uses the whole clip. Checking the same file again uses the same sections.

### 2. Look for patterns in the sound
We use **EfficientAT**, an existing model trained to recognize patterns in audio. It turns each section into a set of numbers describing the sound. We combine these descriptions to get one result for the recording.

### 3. Make a prediction
We trained our own model using examples labeled human-made or AI-generated. It uses the sound description to decide which group your recording is more similar to.

The result is a **score from 0 to 1**: lower leans human, higher leans AI. At **0.5 or above**, the app says **Likely AI-generated**. Below that, it says **Likely human-made**.

### 4. Treat the result as a clue
The model always chooses one of those two labels, even when the score is close to the middle. **A score of 0.9 does not mean there is a 90% chance the song is AI-generated.** The result is a prediction, not proof of who made the music.

### What did it learn from?
We collected **1,000 recordings: 500 labeled human and 500 labeled AI**. The human music came from FMA, a music collection. The AI music came from Echoes TTA and includes music from 12 AI tools.

Our model learned from 700 of those recordings. The other 300 helped us compare versions and check results. We used EfficientAT as it was; we trained the part that makes the human-or-AI prediction.

### What do the charts show?
**Audio overview:** the outlined areas show which parts of your recording we checked.

**Sound map:** shows low and high sounds in the first selected section. The colors show how strong they are. This is a view of the audio, not a map of where AI was detected.

### Can it explain why a song sounds AI-generated?
Not in everyday musical terms. It can give a prediction, but it cannot reliably point to a voice, instrument or moment and say "this is why." The audio charts show what we checked, not evidence that a song was made by AI.

### Where is my recording processed?
The app processes your upload on the computer running it. Uploading a song does not teach or update the model.
"""

INPUT_GUIDANCE = '**10 seconds to 5 minutes, up to 50 MB.** We check up to five 20-second sections from across the recording. Your upload is processed on the computer running this app.'


@st.cache_data(show_spinner=False)
def stylesheet(css_version):
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
    # New widget keys clear previous audio when an example is selected.
    st.session_state['upload_generation'] = st.session_state.get('upload_generation', 0) + 1
    st.session_state['microphone_generation'] = st.session_state.get('microphone_generation', 0) + 1



def responsive_plot(data, caption):
    """Keep plot labels readable on phones with a locally scrollable figure."""
    encoded = base64.b64encode(data).decode('ascii')
    label = html.escape(caption, quote=True)
    st.html(f'<figure class="responsive-plot"><div class="plot-scroll" tabindex="0" role="region" aria-label="{label}">'
            f'<img src="data:image/png;base64,{encoded}" alt="{label}"></div>'
            f'<figcaption>{label}</figcaption></figure>')


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
    st.html('<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable data table"><table class="values-table"><thead><tr>' + head +
            '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


@st.cache_data(show_spinner=False)
def evaluation_artifacts(model_version):
    metrics = json.loads((DEFAULT_OUTPUT / 'metrics.json').read_text(encoding='utf-8'))['metrics']
    predictions = read_csv(DEFAULT_OUTPUT / 'predictions.csv')
    with analysis_lock():
        confusion = figure_png(confusion_figure(metrics))
        distribution = figure_png(distribution_figure(predictions))
    return metrics, predictions, confusion, distribution


def render_analysis():
    left, right = st.columns([1, 1.65], gap='large')
    with left:
        source = st.radio('Add your audio', ['Use microphone', 'Upload a file'],
                          horizontal=True, key='audio_source_mic_first', on_change=clear_result)
        upload = microphone = example = None
        if source == 'Use microphone':
            microphone = st.audio_input('Record nearby music', sample_rate=32000,
                                        key=f'microphone_{st.session_state.get("microphone_generation", 0)}', on_change=upload_changed)
            with st.expander('Microphone not working?'):
                st.write('Allow microphone access in your browser settings. On a phone, open the app using an HTTPS link. A plain HTTP address from your laptop will not enable recording. You can also record with your phone and upload the saved file.')
        else:
            upload = st.file_uploader('Your recording', type=['wav', 'mp3', 'flac', 'ogg'],
                                      key=f'upload_{st.session_state.get("upload_generation", 0)}',
                                      on_change=upload_changed, max_upload_size=50)
        examples, labels = local_examples()
        if examples:
            example = st.selectbox('Try an existing recording, then press Analyze',
                                   options=range(len(examples)), index=None,
                                   format_func=lambda index: labels[index], key='example',
                                   on_change=example_changed)
        if st.button('Analyze recording', type='primary', width='stretch'):
            clear_result()
            if microphone is not None or upload is not None or example is not None:
                with st.spinner('Analyzing recording...'):
                    if microphone is not None:
                        data, name = microphone.getvalue(), 'Microphone recording.wav'
                    elif upload is not None:
                        data, name = upload.getvalue(), upload.name
                    else:
                        path = Path(examples[example][0])
                        data, name = path.read_bytes(), path.name
                    st.session_state['analysis_result'] = run_analysis(data, name)
                    st.session_state['playback_format'] = {'.mp3': 'audio/mpeg', '.flac': 'audio/flac', '.ogg': 'audio/ogg'}.get(Path(name).suffix.lower(), 'audio/wav')
            else:
                if source == 'Use microphone':
                    st.info('Record some music and stop the recording before pressing Analyze.')
                else:
                    st.session_state['analysis_result'] = analyze(None)
        empty_card = '' if st.session_state.get('has_analyzed_audio', False) else EMPTY
        result = st.session_state.get('analysis_result', (empty_card, '', None, None, None, []))
        if result[2] is not None:
            st.session_state['has_analyzed_audio'] = True
            st.divider()
            st.markdown('Listen to the original')
            st.audio(result[2], format=st.session_state.get('playback_format', 'audio/wav'))
            st.markdown(result[1])
    with right:
        if result[0]:
            st.html(result[0])
        if source == 'Use microphone' and result[0] == EMPTY and not st.session_state.get('has_analyzed_audio', False):
            st.caption('Tap record, allow microphone access, and capture 20-30 seconds of music. Stop recording, then press Analyze. At least 10 seconds is required; the limit is 5 minutes.')
            st.caption('Background noise can affect the result. We have not measured accuracy on phone recordings yet.')
        if result[3] is not None:
            responsive_plot(result[3], 'The sections we checked and a sound map of the first section')


def render_results():
    metrics, predictions, confusion, distribution = evaluation_artifacts(hashlib.sha256((DEFAULT_OUTPUT / "metrics.json").read_bytes() + (ROOT / "showcase.py").read_bytes()).hexdigest())
    h = metrics['holdout']
    cm = h['confusion_matrix_true_rows_predicted_columns_human_ai']
    st.markdown(RESULTS_INTRO)
    st.markdown(f'**The model got {cm[0][0] + cm[1][1]} of {h["tracks"]} recordings right ({h["accuracy"]:.1%}) in the final check.** It got {metrics["validation"]["accuracy"]:.1%} right in the earlier check used to choose the model.')
    st.html(f'<div class="stat-grid"><div class="stat"><strong>{cm[1][1]} / {h["ai_tracks"]}</strong><span>AI recordings correctly spotted</span></div><div class="stat"><strong>{cm[0][1]} / {h["human_tracks"]}</strong><span>Human recordings mistaken for AI</span></div><div class="stat"><strong>{h["ai_precision"]:.1%}</strong><span>Of recordings flagged AI, this share had the AI label</span></div></div>')
    responsive_plot(confusion, 'Correct predictions and errors')
    st.markdown(SCORES_INTRO)
    responsive_plot(distribution, 'Scores for human and AI recordings')
    errors = [[r['reference'], r['generator'] or 'Human recording', r['true_label'], r['predicted_label'], float(r['ai_score'])]
              for r in predictions if r['split'] == 'holdout' and r['true_label'] != r['predicted_label']]
    st.markdown('Recordings the model got wrong')
    numeric_table(['Recording', 'Music source', 'Original label', 'Model prediction', 'AI score'], errors)
    st.markdown(LIMITATIONS)


def main():
    st.set_page_config(page_title='AI Music Detector · Audio Lab', layout='wide', initial_sidebar_state='collapsed')
    st.html(stylesheet((ROOT / 'assets/streamlit.css').stat().st_mtime_ns))
    model_version=hashlib.sha256(MODEL_PATH.read_bytes() + Path(__file__).read_bytes() + (ROOT / 'showcase.py').read_bytes()).hexdigest()
    if st.session_state.get('analysis_model_version') != model_version:
        clear_result()
        st.session_state['analysis_model_version']=model_version
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
