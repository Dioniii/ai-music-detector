# AI music detector

An exploratory Python project that classifies direct audio files using ten audio
features and a trained logistic-regression baseline. Historical FMA recordings
provide human reference labels; Echoes TTA provides generated labels. Labels are
research assumptions, not certified authorship.

## Run a prediction

With Python 3.12 and uv, install the locked environment using `uv sync --locked`.
Then, from the repository root:

```powershell
.\.venv\Scripts\python.exe baseline.py predict "path\to\music.wav"
```

The backend analyzes the first ten seconds and returns a binary label and raw AI
score. Scores are not calibrated confidence percentages. Microphone testing,
inconclusive decisions and a Gradio interface are not implemented yet.

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
