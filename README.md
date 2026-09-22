# AI music detector

A beginner portfolio project that extracts ten audio measurements and applies a
trained logistic-regression model. Streamlit shows the prediction, sampled audio
sections, and the contribution of each feature.

**This is an exploratory model, not proof of authorship.** It still makes substantial
human false positives. Its AI score is not a calibrated confidence percentage.

## Run the app

Use Python 3.12. Install the environment once with `uv sync --locked`, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**. Choose an audio file or a local example and press
**Analyze recording**. Accepted uploads are WAV, MP3, FLAC and OGG, from 10 seconds
to 5 minutes and up to 50 MB.

## The six Python files to understand

Read them in this order:

| File | Responsibility |
|---|---|
| `audio_inspection.py` | Load a recording and provide the spectrogram calculation. |
| `preprocessing.py` | Convert to mono at 24 kHz and choose the audio sections. |
| `features.py` | Measure each section and average its ten feature values. |
| `baseline.py` | Train the classifier, load its weights, and predict. |
| `showcase.py` | Turn predictions and evaluation results into charts and display data. |
| `app.py` | Build the Streamlit interface and handle uploads. |

The prediction path is:

`recording -> preprocessing -> ten features -> saved model -> score and explanation`

Sampling uses up to five 20-second sections, one random start in each range of
legal positions. Seed 42 makes the selection repeatable. Shorter recordings use
the available audio; sections may overlap. The first section need not start at
zero. The same selection method is used in training and prediction.

Features summarize RMS amplitude, zero crossings, spectral center, bandwidth and
flatness. Each contributes a mean and a standard deviation. We average the section
vectors, standardize with the saved training statistics, and apply ten learned
weights plus an intercept. A sigmoid converts that sum into an AI score; the
fixed decision threshold is 0.5.

## Predict without the interface

```powershell
.\.venv\Scripts\python.exe baseline.py predict "path\to\recording.mp3"
```

This uses the same model as Streamlit. Its weights and evaluation files live in
`data/baseline_random/demo/`. No paid API or external model service is involved.

## Train the current model again

Training requires the existing 80 local dataset recordings. It does not download
anything. Use a new output folder so the current model is not overwritten:

```powershell
.\.venv\Scripts\python.exe baseline.py train --output data/model_rerun
```

The 20 artist groups stay in the original 12/4/4 split: 48 training recordings,
16 validation and 16 holdout. Only training recordings fit the scaler and model.
The command saves weights, predictions, metrics and sampled sections. To try the
new output from the command line, pass `--model data/model_rerun/model.json`.
Training does not automatically switch the demo to that output.

## What the results mean

The current model detects 10 of 12 AI recordings in the exploratory holdout and
falsely flags 2 of 4 human recordings. These are small, previously inspected
splits, not a fresh final test. Microphone robustness, unfamiliar generators and
an inconclusive outcome have not been established.

Historical FMA recordings supply human-reference labels and Echoes TTA supplies
AI labels. These are research assumptions rather than verified authorship.

## Supporting folders

- `assets/`: dark monospace styling and bundled fonts.
- `data/`: model files, manifests and experiment results. The original audio folders
  are gitignored; local examples appear only when their files exist.
- `tools/`: optional scripts used earlier to source and inspect the dataset.
  They are not needed to run the app. Run them from the repository root as modules,
  for example `python -m tools.dataset_audit --help`.
- `tests/`: existing checks for audio processing and dataset tools.
- `.streamlit/`: theme and upload configuration.

Uploads are processed on the machine hosting the app. Temporary files are deleted
after analysis; playback bytes and results stay in that browser's server session.
Uploaded audio is not put in a shared cache. Streamlit usage telemetry is disabled.

## Project history and hosting

[Deployment instructions](DEPLOYMENT.md) describe the free Community Cloud setup.
The project has not been publicly deployed by this cleanup.

The [journal](journal.md), [first baseline report](data/baseline_v1/REPORT.md),
[volume experiment](data/baseline_v2/REPORT.md), and
[random-sampling report](data/baseline_random/REPORT.md) preserve the findings for
an article. Their old file names and commands describe historical versions. The
current commands and six-file structure are documented above; duplicate experiment
implementations were removed after consolidation. Historical model files are
reference artifacts, not additional supported runtime modes.
