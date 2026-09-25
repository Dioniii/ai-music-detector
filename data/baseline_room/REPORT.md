# Final room-augmentation experiment - 25 September 2026

The augmented classifier is the default Streamlit and CLI model. EfficientAT is unchanged; only the scaler and logistic-regression classifier were fitted. This is one fixed experiment, with no hyperparameter or threshold search.

## Results

Each validation/evaluation set contains 150 recordings: 75 human and 75 AI. Both models use the exact same inputs for each comparison.

| Condition | Split | Model | Accuracy | Human falsely flagged | AI detected |
|---|---|---|---:|---:|---:|
| Clean | Validation | Original encoder classifier | 88.7% | 11/75 | 69/75 |
| Clean | Validation | Augmented classifier | 86.7% | 14/75 | 69/75 |
| Clean | Evaluation | Original encoder classifier | 86.7% | 11/75 | 66/75 |
| Clean | Evaluation | Augmented classifier | 86.7% | 8/75 | 63/75 |
| Simulated room | Validation | Original encoder classifier | 74.0% | 20/75 | 56/75 |
| Simulated room | Validation | Augmented classifier | 77.3% | 14/75 | 55/75 |
| Simulated room | Evaluation | Original encoder classifier | 74.0% | 12/75 | 48/75 |
| Simulated room | Evaluation | Augmented classifier | 78.7% | 10/75 | 53/75 |

## Selection and tradeoffs

Clean validation accuracy loss <=2 percentage points, higher simulated validation accuracy and lower simulated human false-positive rate; no holdout selection.

The candidate passed this rule, using validation only. Clean validation accuracy fell by two percentage points (three additional errors); simulated validation improved from 74.0% to 77.3%, with human false positives falling from 20 to 14. Clean evaluation accuracy stayed at 86.7%, but AI detections fell from 66 to 63 while human false positives fell from 11 to 8. Simulated evaluation improved to 78.7%. These are modest, mixed gains, not a solved microphone detector.

The underlying evaluation music has been reused during development. This is a comparison benchmark, not a fresh blind test. No independent real phone music recordings were available for this run. The user-provided room-silence recording was not used in training or model selection.

## What was trained

- Original dataset: 1,000 recordings, split 700 training / 150 validation / 150 evaluation with the existing artist/reference grouping.
- Fitting uses 700 clean vectors plus exactly 700 altered vectors: 1,400 examples, still only 700 original recordings. Both classes contribute 700 fitting examples each.
- Validation/evaluation variants come only from their own music splits and separate noise source recordings. No augmented evaluation vector fits the scaler or classifier.
- EfficientAT mn10_as stays frozen, producing 960 values. The classifier remains StandardScaler + LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=42), with threshold 0.5.
- One variation is seeded by the recording ID, independent of its label. The same procedure applies to both classes.

## Simulated room processing

Music is converted to mono at 32 kHz, bandpass-filtered (random lower edge 60-220 Hz, upper edge 3.5-12 kHz), and mixed with six short decaying echoes delayed 15-220 ms. Music level varies from -32 to -16 dBFS RMS; noise is mixed at an 8-25 dB signal-to-noise ratio. Peak limiting rescales the whole mixture to avoid deliberate clipping. The unchanged random-section encoder pipeline then extracts its features.

Nine five-second ESC-50 recordings provide rain, washing-machine and breathing sounds, three disjoint original sources per split. Noise loops with a random offset. This is a small environmental-noise bank, not representative cafe chatter, measured room acoustics or a full phone simulation. The files and licenses are documented in [the noise directory](../room_noise/README.md).

Altered audio is temporary. Cached vectors are keyed by decoded audio, encoder, preprocessing/feature source, augmentation source/manifest and seed/noise checksum. No new Python files or packages were added. Inference does not apply augmentation or load noise files.

## Evidence and reproduction

`model.json` holds the trained parameters and augmentation provenance. `features.npz` and `augmented_features.npz` align by candidate ID. `augmentation.json` records each transformation; `simulated_predictions.csv`, `predictions.csv`, `metrics.json` and `comparison.json` preserve the comparisons.

136 existing/updated tests passed. Artifact checks verified all 1,000 rows, the 1,400-vector training-only scaler means/scales, agreement between saved scores and numeric inference, and the original model checksum. Streamlit checks also verified all three tabs, updated benchmark text, example inference, plotting and CLI/UI score parity. The already-recorded recoverable MP3 decoder warning occurred again; the run completed and saved all artifacts.

```powershell
uv run --locked python -m tools.batch_audio download-expanded
uv run --locked python -m tools.batch_audio download-room-noise
uv run --locked baseline.py train --encoder --augment --output data/room_rerun
```

Training always writes a new directory; it never automatically replaces the app model. The original classifier remains at `data/baseline_encoder/model.json`. No public deployment or Git push was performed by this experiment.
