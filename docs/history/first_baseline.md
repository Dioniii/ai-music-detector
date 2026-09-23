> Historical snapshot. For the current setup and file layout, see the [project README](../../README.md). Earlier raw experiment outputs are preserved in [the archive](../../data/archive/early_experiments.zip), using their original paths. Audio and plot links may refer to local-only files.

# First trained baseline

Run date: 2026-09-22. This is an exploratory portfolio experiment, not a validated
universal detector. We deliberately moved to training without another standalone
quality-review stage. Pending audio/source review flags remain documented.

## Model and experiment

The existing first-ten-second baseline features form an `(80, 10)` matrix. Only
the ten FEATURE_NAMES values enter the model; labels, generator names and artist
IDs do not. We reuse the original preprocessing, without the diagnostic mean removal.

A fixed hash order (`baseline-v1:` plus artist group ID) assigns 12 artist groups
to training, four to validation and four to holdout. The known pilot artist group
is forced into training first. That produces 48/16/16 recording rows. References
and artists stay within one split; existing group IDs are still provisional with
near-duplicate/alias review pending. These experimental assignments are saved in
predictions.csv; source eligibility and manifest fields remain unchanged.

The scikit-learn Pipeline fits StandardScaler and logistic regression on the 48
training rows only. Scaling subtracts each training feature mean and divides by its
training standard deviation. Logistic regression learns ten weights and an intercept.
`class_weight=balanced` compensates for the training count of 12 human and 36 AI rows.
Settings: C=1, lbfgs, maximum 1,000 iterations, random_state=42. The operating
threshold is fixed at 0.5, not selected from the holdout. No hyperparameter search,
threshold tuning or calibration was performed; validation reports behavior of the
fixed model. All three generators occur in each split; this is not an unseen-generator
experiment. Original files and preprocessing code remain unchanged.

## Results

| Partition | Groups | Human / AI | AI precision | AI recall | Human false-positive rate | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| train | 12 | 12 / 36 | 96.88% | 86.11% | 8.33% | 87.50% |
| validation | 4 | 4 / 12 | 78.57% | 91.67% | 75.00% | 75.00% |
| holdout | 4 | 4 / 12 | 84.62% | 91.67% | 50.00% | 81.25% |

Holdout confusion matrix (rows are true labels, columns predictions):

| True label | Predicted human | Predicted AI |
|---|---:|---:|
| Human | 2 | 2 |
| AI | 1 | 11 |

The model detects 11 of 12 AI-labeled examples, but falsely labels two of four human
references as AI. Validation also has three false positives among four humans.
That is a substantial weakness, not a production-ready result. Always predicting AI
would already yield 75% accuracy on this 3:1 class mix; the observed 81.25% holdout
accuracy must not be presented without the false-positive counts. With only four
human holdout examples, each error changes human FPR by 25 percentage points.

This batch was previously inspected and used for preprocessing diagnostics. The
holdout is separated from parameter fitting, but it is not an untouched final test
of the overall development process. Dataset-label uncertainty, source encoding,
bandwidth, offsets, genre imbalance and unresolved listening/near-duplicate review
limit interpretation. We did not repair these issues or silently certify eligibility.
Microphone robustness and calibrated/inconclusive outcomes remain future work.

### Holdout errors

- The Dawn - The One They Fear (audioldm): true ai, predicted human, AI score 0.2364.
- Robot Heart - Mystery Mammal (human reference): true human, predicted ai, AI score 0.8739.
- Vanishing Horizon - Jason Shaw (human reference): true human, predicted ai, AI score 0.6071.

## Working inference and artifacts

`baseline.py predict` accepts a decodable local audio file containing at least ten
seconds. It extracts the same first-ten-second representation and applies the
saved scaler, weights, intercept and threshold. An AI score is a raw model output,
not a calibrated probability of authorship. This first binary baseline has no
inconclusive rule and should not force product-level certainty.

- [model.json](../../data/archive/early_experiments.zip): portable numeric parameters, feature order, settings,
  scikit-learn version and preprocessing-source hashes.
- [metrics.json](../../data/archive/early_experiments.zip): actual partition metrics, policies and input hashes.
- [predictions.csv](../../data/archive/early_experiments.zip): all 80 assignments, labels and scores,
  including mistakes for later review.

```powershell
.\.venv\Scripts\python.exe baseline.py predict "path\to\music.wav"
.\.venv\Scripts\python.exe baseline.py train --output data/baseline_v1_rerun
```

Training refuses to overwrite an existing experiment. Prediction rejects changed
audio/feature source hashes so a changed preprocessing implementation cannot silently
reuse this model. It loads numeric JSON rather than executable pickle objects.

## Verification performed

As requested, no extra test suite was added and no standalone testing stage was
introduced. During the actual run we checked group separation, both classes in
all partitions, finite `(80, 10)` features, convergence and label consistency.
The exported numeric predictor matched scikit-learn scores across all 80 rows to
1e-12 tolerance. Fresh extraction from a real audio file matched its cached feature
vector to 1e-10 tolerance. Saved-model local-file inference completed successfully.
That example was itself a false positive, which illustrates why operational success
and model usefulness are different questions.

scikit-learn 1.9.1 was already installed and locked transitively. It is now a direct
dependency. Offline full resolution lacked cached metadata; retaining the existing
package entry and updating the root dependency metadata passed `uv lock --check
--offline`. No packages or data were downloaded. The existing unit suite was not
rerun for this increment; verification focused on the actual training/inference path.

## Portfolio takeaway

We now have a reproducible trained baseline and file-to-prediction backend, with
transparent failure cases. A fair next comparison can measure whether a frozen
audio encoder improves human false positives over these handcrafted features.
That is future work, not a result or additional implementation in this increment.

Prepared with Codex assistance. No model accuracy, listening results or authorship
certification are inferred beyond the recorded experiment.
