# AI music detector

A beginner portfolio project that uses a frozen EfficientAT audio encoder and
our own trained logistic-regression classifier. Streamlit shows the prediction,
sampled audio sections and the strongest learned-dimension contributions.

**This is an exploratory model, not proof of authorship.** Its AI score is not
a calibrated confidence percentage. The earlier ten-feature baseline is retained
for comparison.

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
| `features.py` | Extract and average EfficientAT embeddings, or historical handcrafted features. |
| `baseline.py` | Train the classifier, load its weights, and predict. |
| `showcase.py` | Turn predictions and evaluation results into charts and display data. |
| `app.py` | Build the Streamlit interface and handle uploads. |

The prediction path is:

`recording -> 32 kHz sections -> frozen encoder -> 960 features -> trained classifier -> score`

Sampling uses up to five 20-second sections, one random start in each range of
legal positions. Seed 42 makes the selection repeatable. Shorter recordings use
the available audio; sections may overlap. The first section need not start at
zero. The same selection method is used in training and prediction.

EfficientAT converts each section into 960 learned features. We average those
vectors, standardize with training-only statistics and apply our own learned
classifier weights. The decision threshold stays at 0.5. Embedding dimensions do
not have simple physical meanings; the chart shows their arithmetic contribution.

## Predict without the interface

```powershell
.\.venv\Scripts\python.exe baseline.py predict "path\to\recording.mp3"
```

This uses the same model as Streamlit. Its weights and evaluation files live in
`data/baseline_encoder/`; the frozen encoder is in `data/encoder/`. No paid API or external model service is involved.

## Train on the 1,000-recording dataset

The fixed manifest is `data/dataset_1000.csv`: 500 human-reference recordings
from FMA Small and 500 generated recordings from Echoes TTA, covering 12 generators.
To download only the selected recordings (about 1.6 GiB), then train:

```powershell
.\.venv\Scripts\python.exe -m tools.batch_audio download-expanded
.\.venv\Scripts\python.exe baseline.py train --encoder --output data/model_rerun
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
by omitting `--encoder` and using `--manifest data/batch_manifest.csv`.

## Current encoder results

Validation accuracy is **88.7%**, compared with 68.0% for the previous demo on
the same validation recordings. On the reused 150-recording evaluation set,
accuracy is **86.7%**, AI detection is **66/75**, and human false positives are
**11/75**. This is a reused benchmark, not a new blind test. Dea falls on the
human side, but that known example was not used for model selection.

[Encoder experiment and implementation report](data/baseline_encoder/REPORT.md)
explains the method, results and checks. [Encoder packaging](data/encoder/README.md)
records the source, license, export procedure and local CPU benchmark.
The [handcrafted 1,000-recording experiment](data/baseline_1000/REPORT.md) remains
as a historical comparison. Try the old demo with
`--model data/baseline_random/demo/model.json`.

Microphone robustness and unfamiliar-generator performance remain untested.
Historical FMA supplies human-reference labels and Echoes TTA supplies generated
labels; these are dataset assumptions rather than verified authorship.

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
