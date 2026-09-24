# AI Music Detector

An audio classification project that explores whether a recording is human-made
or AI-generated. It combines a pretrained **EfficientAT** encoder with a
**logistic-regression classifier trained for this project**, presented through
an interactive Streamlit app.

Upload a song, inspect the sampled audio, and see how the classifier reached its
result. The output is an experimental model score, not proof of authorship.

## The demo

- **Analyze a recording** and get a human/AI prediction with its model score.
- **See what was sampled** through a waveform and a spectrogram.
- **Inspect the classifier** through the strongest embedding contributions.
- **Explore the results**, including incorrectly classified recordings.

The interface uses a dark monospace design and supports desktop and mobile
screens. Inference runs on the app server using CPU PyTorch, without a paid API.

## How it works

```text
Audio -> sampled sections -> EfficientAT embeddings -> trained classifier -> AI score
```

Audio is converted to mono at 32 kHz. Up to five 20-second sections are selected
across the recording using a fixed random seed, making repeated analyses
consistent. Shorter files use the available audio; sections can overlap.

EfficientAT describes each section with **960 learned values**, called an
embedding. Their average is standardized and passed to the classifier. The
encoder stays frozen; the scaler and classifier learn from the project's
labeled training recordings.

Scores at or above **0.5** receive the AI label. The contribution chart shows
how embedding values influence that score. Individual dimensions are not named
musical properties such as vocals or loudness.

## Results

The current dataset contains **1,000 recordings**: 500 human references from
FMA Small and 500 generated recordings from Echoes TTA, covering 12 generators.
The split is **700 training / 150 validation / 150 evaluation**, with balanced
classes and related reference artists kept together.

The project began with ten handcrafted audio measurements. Expanding the dataset
did not clearly improve that approach; replacing those measurements with encoder
embeddings produced a stronger result.

| Approach | Training recordings | Validation accuracy | Evaluation accuracy |
|---|---:|---:|---:|
| Initial handcrafted baseline | 48 | 68.0% | 61.3% |
| Expanded handcrafted baseline | 700 | 63.3% | 63.3% |
| **EfficientAT + logistic regression** | **700** | **88.7%** | **86.7%** |

All rows use the same 150 validation and 150 evaluation recordings. The encoder
model detected **66 of 75 AI recordings** and incorrectly flagged **11 of 75
human recordings** in evaluation.

These evaluation recordings were reused across experiments, so this is a
comparison benchmark rather than a fresh blind test. Full methodology and
limitations are in the [experiment report](data/baseline_encoder/REPORT.md).

## Run locally

With **Python 3.12** and **uv** installed, run these commands from the repository
root:

```bash
uv sync --locked
uv run --locked streamlit run app.py
```

Open **http://localhost:8501**. Upload a WAV, MP3, FLAC or OGG file between
**10 seconds and 5 minutes**, up to **50 MB**. Local examples appear when dataset
recordings are available on your machine.

The encoder and classifier are included. You do not need to download the training
dataset to use the app. Uploaded temporary files are deleted after analysis;
playback and results remain in the current browser's server session.

To predict from the command line:

```bash
uv run --locked baseline.py predict "path/to/recording.mp3"
```

<details>
<summary><strong>Reproduce training</strong></summary>

Download the fixed selection of recordings, then train into a new output folder:

```bash
uv run --locked python -m tools.batch_audio download-expanded
uv run --locked baseline.py train --encoder --output data/encoder_rerun
```

The download is approximately 1.6 GiB and resumes completed files. Audio and
embedding caches stay local. Only training recordings fit the scaler and
classifier; the saved manifest fixes the splits.

Training saves a new model and evaluation artifacts without replacing the demo.
To use that model, pass `--model data/encoder_rerun/model.json` to the prediction
command. The earlier baseline remains available at
`data/baseline_random/demo/model.json`.

Run the existing checks with:

```bash
uv run --locked pytest -q
```

</details>

## Inside the project

| Files | Purpose |
|---|---|
| `audio_inspection.py`, `preprocessing.py` | Load audio and select sections. |
| `features.py`, `baseline.py` | Extract embeddings, train and predict. |
| `showcase.py`, `app.py` | Generate charts and build the interface. |

`data/` holds model artifacts, manifests and experiment results. `tools/` contains
dataset preparation commands, and `tests/` contains checks. Downloaded audio
and personal recordings are gitignored.

## Scope and limitations

This is an exploratory detector. Its score is not a calibrated probability,
and dataset labels are assumptions rather than verified authorship. The current
evaluation covers Rock and Electronic, with no Pop. Microphone recordings,
coffee-shop noise and unfamiliar generators have not been validated. Source
encoding and other dataset differences may influence predictions.

Public deployment and Streamlit Community Cloud resource usage remain unverified.

## Further reading

- [Experiment report](data/baseline_encoder/REPORT.md) - model comparison and verification.
- [Encoder details](data/encoder/README.md) - source, attribution and local CPU benchmark.
