# AI Music Detector

An audio classification project that explores whether a recording is human-made
or AI-generated. It combines a pretrained **EfficientAT** encoder with a
**logistic-regression classifier trained for this project**, presented through
an interactive Streamlit app.

Record nearby music or upload a song, inspect the sampled audio, and view the
model result. The output is an experimental model score, not proof of authorship.

## The demo

- **Analyze a recording** and get a human/AI prediction with its model score.
- **See what was sampled** through a waveform and a spectrogram.
- **Record again when needed** with a basic weak-signal and clipping check.
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

Scores at or above **0.5** receive the AI label. The model cannot explain its
result in terms of named musical properties such as vocals or instruments.
The UI rejects recordings with fewer than five seconds above its minimum level
or heavy clipping; this is not a reliable detector of music over room noise.

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
| Original EfficientAT classifier | 700 | 88.7% | 86.7% |
| **Final classifier with room augmentation** | **700 + 700 altered copies** | **86.7%** | **86.7%** |

All rows use the same 150 validation and 150 evaluation recordings. The final
model detected **63 of 75 AI recordings** and incorrectly flagged **8 of 75
human recordings** on clean evaluation. The original encoder detected 66 and
flagged 11: fewer human false positives now comes with more missed AI recordings.

The final experiment added one deterministic room/noise variation per training
recording, keeping EfficientAT frozen. On the same 150 evaluation recordings
with simulated room effects, accuracy improved from **74.0% to 78.7%**. Human
false positives fell from **12/75 to 10/75**, and AI detections rose from **48/75
to 53/75**. Separate noise source recordings were used for training, validation
and evaluation. This does not establish accuracy on actual phone recordings.

The candidate met the rule set before training: no more than a two-point clean
validation accuracy drop, improved simulated validation accuracy and fewer
simulated validation human false positives. No threshold or parameter search
was performed. This is the final planned model experiment.

These evaluation recordings were reused across experiments, so this is a
comparison benchmark rather than a fresh blind test. Full methodology and
limitations are in the [experiment report](data/baseline_room/REPORT.md).

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
uv run --locked python -m tools.batch_audio download-room-noise
uv run --locked baseline.py train --encoder --augment --output data/room_rerun
```

The download is approximately 1.6 GiB and resumes completed files. Audio and
embedding caches stay local. Only training recordings fit the scaler and
classifier; the saved manifest fixes the splits.

Training saves a new model and evaluation artifacts without replacing the demo.
To use that model, pass `--model data/room_rerun/model.json` to the prediction
command. The original encoder model remains at `data/baseline_encoder/model.json`; the
earlier handcrafted baseline remains at `data/baseline_random/demo/model.json`.
The nine attributed noise clips add about 3.8 MiB of downloads and remain local.
See [noise sources and licensing](data/room_noise/README.md).

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

- [Experiment report](data/baseline_room/REPORT.md) - model comparison and verification.
- [Encoder details](data/encoder/README.md) - source, attribution and local CPU benchmark.
