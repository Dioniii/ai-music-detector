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
| `preprocessing.py` | Convert to mono at 32 kHz for the encoder (24 kHz for the baseline) and choose sections. |
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
predictions, metrics, sampled sections and a comparison with the preserved handcrafted baseline.

To try another output from the command line, pass
`--model data/model_rerun/model.json`. Training does not automatically switch
the demo to that output. Historical 80-recording training is still available
by omitting `--encoder` and using `--manifest data/preparation/batch_manifest.csv`.

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

## Repository layout

```text
app.py, showcase.py                 Streamlit interface and charts
baseline.py, features.py            Training, prediction and encoder
preprocessing.py, audio_inspection.py  Audio loading and sampling
assets/                             Styling and fonts
.streamlit/                         App configuration
data/
  encoder/                          Frozen encoder, license and provenance
  baseline_encoder/                 Current classifier and evaluation
  baseline_1000/                    Handcrafted comparison and reusable features
  baseline_random/demo/             Preserved earlier baseline
  dataset_1000.csv                   Current dataset selection and splits
  dataset_1000_plan.json             Dataset selection summary
  preparation/                      Earlier manifests and preparation records
  archive/                          One ZIP of early raw experiment outputs
docs/
  DEPLOYMENT.md                     Hosting instructions
  journal.md                        Project decisions and progress
  history/                          Earlier reports and learning debriefs
tools/                              Dataset download and preparation commands
tests/                              Audio and dataset checks
```

Downloaded audio, source metadata, embedding caches and personal recordings are
gitignored. Local examples appear only when their files exist. Personal test
recordings belong in `data/local_recordings/`.

The five scripts in `tools/` support dataset preparation and download; the app
does not import them. Run them from the repository root, for example
`python -m tools.batch_audio --help`. They remain separate from the six main
Python files so the normal learning path stays short.

Uploads are processed on the machine hosting the app. Temporary files are deleted
after analysis; playback bytes and results stay in that browser's server session.
Uploaded audio is not put in a shared cache. Streamlit usage telemetry is disabled.

## Project history and hosting

[Deployment instructions](docs/DEPLOYMENT.md) describe the free Community Cloud setup.
Public deployment remains a separate step.

The [journal](docs/journal.md), [first baseline report](docs/history/first_baseline.md),
[volume experiment](docs/history/volume_experiment.md), and
[random-sampling report](docs/history/random_sampling.md) preserve the findings for
an article. These are historical snapshots, not current run instructions.
The [early experiment archive](data/archive/early_experiments.zip) replaces 28
loose generated files. It preserves their original repository paths and includes
an `ARCHIVE_INDEX.json` with SHA-256 checksums. Open it to recover old raw results;
it is not required to run or train the current model.
