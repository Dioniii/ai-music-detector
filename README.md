# AI music detector

An exploratory Python project that classifies direct audio files using ten audio
features and a trained logistic-regression baseline. Historical FMA recordings
provide human reference labels; Echoes TTA provides generated labels. Labels are
research assumptions, not certified authorship.

## Open the Streamlit demo

Install the environment with `uv sync --locked`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Upload a WAV, MP3, FLAC or OGG recording (10 seconds
to 5 minutes, up to 50 MB), or select a local example immediately below the uploader
and press **Analyze recording**. The three tabs retain the dark IBM Plex Mono
interface, sharp borders, teal/amber results, and clear primary action.

The demo includes original-audio playback, waveform and spectrogram, raw score,
exact feature contributions, and the saved model's confusion matrices and errors.
Changing the recording clears the previous result. Short or invalid files produce
a readable message.

Audio processing runs wherever the app is hosted: locally during development,
on the server after cloud deployment. Each upload is decoded through an isolated
temporary file that is deleted after analysis. Audio and results are held in the
browser's server session, never the shared application cache. No paid inference
API, encoder download, or external model service is involved. Streamlit usage
telemetry is disabled. Existing recordings are offered only when their local
files exist; dataset audio is not bundled or downloaded at startup.

To use another port, append `--server.port 8502`. To restrict access to this
computer, append `--server.address 127.0.0.1`.

See [the Streamlit migration debrief](STREAMLIT_DEBRIEF.md) for implementation
and verification, and [cloud deployment instructions](DEPLOYMENT.md) for the
free Community Cloud setup. The [Gradio debrief](GRADIO_DEBRIEF.md) remains a
historical record of the previous interface.

## Run a prediction

With Python 3.12 and uv, install the locked environment using `uv sync --locked`.
Then, from the repository root:

```powershell
.\.venv\Scripts\python.exe baseline.py predict "path\to\music.wav"
```

The backend analyzes the first ten seconds and returns a binary label and raw AI
score. Scores are not calibrated confidence percentages. Microphone testing and
inconclusive decisions are not implemented yet.

## First measured baseline

The exploratory holdout contains four artist groups: four human and 12 generated
recordings. The model detected 11 AI examples and falsely flagged two human ones:
84.6% AI precision, 91.7% AI recall, and 50% human false-positive rate. This small,
previously inspected batch is not an untouched final test. The false positives
make the current model unsuitable for confident authorship claims.

See the [experiment report](data/baseline_v1/REPORT.md) for settings, confusion
matrices, limitations and errors. [Predictions](data/baseline_v1/predictions.csv)
and [model parameters](data/baseline_v1/model.json) are saved for reproducibility.

## How it works

`audio_inspection.py` loads audio; `preprocessing.py` converts it to a ten-second,
24 kHz mono clip; `features.py` produces ten measurements; `baseline.py` fits
training-only scaling and logistic regression, evaluates group-separated data,
and predicts from local files. Original audio remains local and gitignored.

To reproduce training when the batch feature artifacts are present:

```powershell
.\.venv\Scripts\python.exe baseline.py train --output data/baseline_v1_rerun
```

The saved paired diagnostic CSV supplies only its unchanged-preprocessing
`baseline` rows. The comparison/centered rows are not used. No additional download
is performed by training. One local batch recording is needed for its feature
round-trip check; source audio is not redistributed with this repository.

[Engineering journal](journal.md) records progress and decisions. Dataset sourcing
and attribution are documented in the [audit](data/audit.md), [batch report](data/batch_audio_report.md)
and manifests. Built with NumPy, SciPy, librosa, scikit-learn and Codex assistance.
