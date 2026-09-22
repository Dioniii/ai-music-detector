# Reproducible random-section sampling

## Implemented behavior

The Streamlit demo now uses the validation-selected multi_raw model trained with stratified random section selection. This is a sampling change, not a demonstrated accuracy improvement. Earlier v1 and evenly spaced v2 models and their code remain intact.

1. Decode and resample to 24 kHz mono.
2. Use sections of 20 seconds, or the whole recording when it is 10-20 seconds long.
3. Use min(5, ceil(duration / 20)) sections, capped by the number of legal starts.
4. Partition legal sample start positions into equally sized, disjoint ranges.
5. Draw one integer start from each range using NumPy PCG64 seed 42.
6. Average the section feature vectors; apply the scaler and classifier trained using exactly this sampling procedure.

The beginning and end are not forced into the selection. Selected sections may overlap. No silence, genre, predicted label, or other content is used to select a favorable segment. Identical resampled durations share start positions; this is intentional, and keeps repeats, renamed copies and volume changes aligned. It is deterministic pseudorandom sampling, not a fresh draw on each click.

## Training and selection

The original artist-group splits and fixed 0.5 threshold are unchanged. Training fits scalers and classifiers only on 48 training recordings. The raw, no-RMS, and RMS-normalized feature treatments were compared on the existing validation split. Highest validation balanced accuracy selects the raw variant. No external-song result was used for selection.

| Variant | Validation human false positives | Validation AI detected | Old holdout human false positives | Old holdout AI detected |
|---|---:|---:|---:|---:|
| v1_first10 | 3/4 | 11/12 | 2/4 | 11/12 |
| multi_raw | 2/4 | 11/12 | 2/4 | 10/12 |
| multi_no_rms | 2/4 | 10/12 | 2/4 | 8/12 |
| multi_rms_normalized | 3/4 | 10/12 | 2/4 | 11/12 |

The deployed local model still has a 50% human false-positive rate on four old holdout human tracks. It detects 10/12 AI examples there, versus 11/12 for v1. The repeatedly inspected holdout is exploratory, and these results do not establish an improvement.

## Dea diagnostic

The selected model returns an AI score of 0.797715, still a false positive under the human label supplied by the user. Repeated runs produced identical features, timestamps and scores. Sampled intervals:

- 3.77-23.77 seconds.
- 74.84-94.84 seconds.
- 111.99-131.99 seconds.
- 145.08-165.08 seconds.
- 187.03-207.03 seconds.

## Demo integration

showcase_random.py adapts the selected model to the existing interface. The waveform outlines every selected section, the spectrogram shows the first sampled section, and the detail text lists all timestamps. The feature table and contributions describe the averaged recording-level features. Example labels and evaluation plots are regenerated from the new predictions, not reused v1 predictions.

The data/baseline_random/demo directory is a snapshot of the validation-selected model plus metrics and predictions in the UI display format. Cached evaluation data is keyed by its file digest. Changing model parameters clears an existing session result, preventing old scores being shown under a new model.

## Reproduce

```powershell
.\.venv\Scripts\python.exe baseline_random.py predict "path\to\recording.mp3"
.\.venv\Scripts\python.exe baseline_random.py train --output data/baseline_random_rerun
```

The prediction command defaults to the variant named in comparison.json. Training a new output directory never overwrites the existing models or automatically publishes a new demo snapshot. All original audio must be available to rerun training; inference only needs the model and code.

## Verification

Checked deterministic starts, unique legal positions and one selection per range at 10, 20, 21, 30, 120 and 230 seconds. Verified exact repeatability on Dea, JSON/sklearn score parity during training, and Streamlit example analysis with timestamps, both figures, ten contribution rows and CLI score parity. Source-hash guards protect the saved random models. No public deployment or external audio upload was performed.
