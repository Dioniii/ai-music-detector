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

## Train on the 1,000-recording dataset

The fixed manifest is `data/dataset_1000.csv`: 500 human-reference recordings
from FMA Small and 500 generated recordings from Echoes TTA, covering 12 generators.
To download only the selected recordings (about 1.6 GiB), then train:

```powershell
.\.venv\Scripts\python.exe -m tools.batch_audio download-expanded
.\.venv\Scripts\python.exe baseline.py train --output data/model_rerun
```

Downloads resume using verified local receipts. Training itself does not download
anything. Use a new output folder to preserve previous results.

The fixed split is **700 training, 150 validation and 150 holdout**, balanced
between the two labels. Related artists and reference recordings stay together;
previously inspected artists are restricted to training. Only training data fits
the scaler and classifier. The command saves weights, extracted features,
predictions, metrics, sampled sections and a comparison with the demo model.

To try another output from the command line, pass
`--model data/model_rerun/model.json`. Training does not automatically switch
the demo to that output. Historical 80-recording training is still available
with `--manifest data/batch_manifest.csv`.

## The 1,000-recording experiment

The larger dataset is downloaded and the new model is saved in
`data/baseline_1000/`. On the same new 150-recording holdout, human false positives
fell from **25/75 to 21/75**, while AI detections fell from **42/75 to 41/75**.
Validation accuracy declined from **68.0% to 63.3%**, so the demo still uses the
previous model. Dea now falls on the human side with the new model, but that
single known example does not justify replacing the demo model.

Try the experimental model with
`python baseline.py predict "path/to/recording.mp3" --model data/baseline_1000/model.json`.
The [experiment report](data/baseline_1000/REPORT.md) explains the data, method,
comparison, limitations and commands in detail.

## What the current demo results mean

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
