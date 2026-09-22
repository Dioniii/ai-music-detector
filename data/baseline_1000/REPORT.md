# 1,000-recording experiment

## Outcome

Downloaded the complete 500-human / 500-AI dataset and trained the existing
classifier on its 700 training recordings. The remaining 150 validation and
150 holdout recordings were excluded from fitting.

**The new model is saved, but it does not replace the Streamlit demo model.**
The pre-recorded decision rule was to use validation accuracy, which decreased
from 68.0% to 63.3%. Validation has balanced classes, so this is also balanced
accuracy. The fixed 0.5 threshold and model settings were not tuned on holdout.

On the new holdout, the new model reduced human false positives by four, missed
one more AI recording, and improved accuracy by two percentage points. That is
a mixed result, not evidence of reliable authorship detection. Both models
below were evaluated on exactly the same new recordings.

| Split | Model | Accuracy | Human falsely flagged | AI detected |
|---|---|---:|---:|---:|
| train | Previous model | 65.6% | 114/350 | 223/350 |
| train | 1,000-recording experiment | 69.1% | 104/350 | 238/350 |
| validation | Previous model | 68.0% | 25/75 | 52/75 |
| validation | 1,000-recording experiment | 63.3% | 25/75 | 45/75 |
| holdout | Previous model | 61.3% | 25/75 | 42/75 |
| holdout | 1,000-recording experiment | 63.3% | 21/75 | 41/75 |

Dea, the previously inspected user recording, changed from AI score **0.797715**
to **0.458691**, crossing to the human side of the 0.5 threshold. This recording
was not used for fitting or model selection. It is a known diagnostic example,
not a blind test. The new score is near the threshold and is not a calibrated
probability. See diagnostics.json.

## What changed and why

The earlier batch had only 80 recordings and three generators. This experiment
uses 1,000 recordings and 12 generators to see whether more varied training
examples help the same simple model. No encoder, neural network, augmentation,
volume normalization or additional Python module was introduced.

Human-reference recordings come from FMA Small; generated recordings come from
Echoes TTA. Selection uses metadata and deterministic SHA-256 ordering, never
model predictions. Pop, Rock and Electronic counts match between classes within
each split. AI generator counts range from 40 to 44 across the whole dataset.

| Split | Human | AI | All artist groups | AI reference-artist groups |
|---|---:|---:|---:|---:|
| Training | 350 | 350 | 350 | 29 |
| Validation | 75 | 75 | 75 | 7 |
| Holdout | 75 | 75 | 78 | 9 |

Artists and related reference recordings remain in one split. Previously
inspected artists are restricted to training. The 75 AI recordings in a test
split do not represent 75 independent reference artists. All 12 generators
appear across splits; this is not an unseen-generator experiment.

## How the code trains it

1. `tools/batch_audio.py` selects entries and downloads individual ZIP members.
   The checked-in manifest contains source URLs, archive members, label and
   license metadata, artist/reference IDs, splits and local paths. The source
   metadata caches are needed only to recreate selection; the manifest is
   sufficient to download this fixed dataset.
2. `audio_inspection.py` decodes each recording. `preprocessing.py` averages
   channels, resamples to 24 kHz, and selects up to five 20-second sections with
   fixed-seed stratified random sampling. Shorter recordings use available audio.
3. `features.py` measures RMS, zero crossings, spectral center, bandwidth and
   flatness, using a mean and standard deviation for each. It averages section
   vectors into ten numbers per recording.
4. `baseline.py` fits StandardScaler on the 700 training vectors, then fits
   scikit-learn LogisticRegression with C=1, balanced class weights, max_iter=1000
   and random_state=42. It exports the scaler, ten weights and intercept to JSON.
5. It scores each split and the previous model on the same vectors. An AI score
   at or above 0.5 produces the AI label. Validation decides promotion; holdout
   describes performance without selecting settings.

## Limitations recorded during this run

- The holdout has Electronic and Rock, with no Pop; validation has all three.
  Its metrics do not establish Pop performance.
- Historical FMA and Echoes labels are dataset assumptions. Source encoding,
  duration and bandwidth can still be shortcuts. FMA files are about 30 seconds,
  whereas AI recording lengths vary; nine selected AI files exceed five minutes.
  The app's five-minute upload limit is unchanged.
- Exact decoded duplicates: zero. This is not a near-duplicate or listening audit.
- FMA track 29245, "The Angel - Benjamin Bret", emitted an mpg123 dequantization
  warning but decoded to 30.0027 seconds of finite, nonzero audio. It was retained
  in training and recorded as a source-quality limitation.
- Microphone, room-noise and unfamiliar-generator performance remain untested.
- FMA license metadata includes CC BY, BY-SA, BY-NC and BY-NC-SA; ND entries were
  excluded. Individual license URLs remain in the manifest. Downloaded audio
  stays gitignored and is not included in the public app.

## Artifacts and reproduction

- `../dataset_1000.csv`: fixed recording selection and splits.
- `../dataset_1000_plan.json`: selection counts and payload estimate.
- `model.json`: new portable numeric model, including training-manifest hash.
- `features.csv`: the ten extracted values for every recording.
- `sections.json`: sampled timestamps, durations and exact decoded-audio hashes.
- `metrics.json`, `predictions.csv`: aggregate and per-recording evaluation.
- `comparison.json`: both models on the same splits, previous-model hash and decision.
- `diagnostics.json`: the previously inspected Dea example.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m tools.batch_audio download-expanded
.\.venv\Scripts\python.exe baseline.py train --output data/model_rerun
.\.venv\Scripts\python.exe baseline.py predict "C:\Users\38349\Downloads\TDS - Dea (Official Video HD).mp3" --model data/baseline_1000/model.json
```

Downloads resume with hashes/receipts and bounded timeout retries. The actual
response-body transfer, including resumed index requests and partial retries,
was 1,652,685,473 bytes (1.539 GiB).
The old demo model is preserved at `data/baseline_random/demo/`.
No paid service, commit, push or public deployment was performed.

## Interpretation for the portfolio

This experiment demonstrates a complete data-to-evaluation workflow and an
honest model comparison. More data did not produce a clear overall improvement.
Training accuracy is also modest (69.1%), which suggests examining the current
features and representation before simply increasing the dataset again.
This result alone does not prove which next model will work better.

Verification: all 121 existing tests passed (one pre-existing short-spectrogram
warning). The exported model reproduced all 1,000 saved scores within 1e-12;
its scaler mean and scale matched only the 700 training feature vectors. The
manifest hash matched, all three splits contained both classes and all 12
generators, and the previous demo model hash remained unchanged. The repository
still has six root Python modules. The new model JSON is 1,698 bytes.
