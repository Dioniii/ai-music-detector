# Longer-audio and volume-sensitivity experiment

## Outcome

Three new classifiers were trained on the same 48 training recordings. No model was promoted into Streamlit. Longer sampling and volume treatment did not resolve the known Dea false positive.

## Audio contract

- Mono at 24 kHz, preserving the original resampling filter.
- Up to five evenly spaced sections, each up to 20 seconds, including the beginning and end.
- 10-20 second recordings use the entire available signal. Longer recordings use min(5, ceil(duration / 20)) sections.
- A 30-second recording uses sections 0-20 and 10-30 seconds; overlap is intentional.
- Long songs contribute up to 100 seconds, not their complete duration.
- Each section produces the same ten frame-summary measurements as v1; section vectors are averaged into one recording vector. This averages features, not probabilities.
- Normalization sets each non-silent section to RMS amplitude 0.1; it is not LUFS/perceptual loudness normalization. No peak clipping is applied. Near-silent sections (RMS <= 1e-8) are left unchanged.

## Fair comparison

The original 12/4/4 artist-group assignment is unchanged. Scalers and classifiers fit training data only. Every model uses balanced logistic regression, C=1 and a fixed 0.5 decision threshold. No external song was used to train, select, or tune a model. The old holdout was already inspected, so these are exploratory comparisons, not a fresh generalization estimate.

| Version | Validation human false positives | Validation AI detected | Old holdout human false positives | Old holdout AI detected |
|---|---:|---:|---:|---:|
| v1_first10 | 3/4 | 11/12 | 2/4 | 11/12 |
| multi_raw | 2/4 | 12/12 | 2/4 | 10/12 |
| multi_no_rms | 2/4 | 10/12 | 2/4 | 8/12 |
| multi_rms_normalized | 3/4 | 11/12 | 1/4 | 10/12 |

The predeclared validation rule selected **multi_raw** (highest balanced accuracy, then fewer human false positives). It performed worse than v1 on the old holdout. The normalized variant did better on the old holdout but not validation; selecting it based on that holdout would reuse evaluation data for model selection. These small splits do not support a reliable winner.

## Dea diagnostic

The user identifies this direct MP3 as human-made. This is an inspected diagnostic example, not an independently verified blind test. Its duration is 230.946 seconds. Sections start at approximately 0, 52.736, 105.473, 158.209 and 210.946 seconds, each lasting 20 seconds.

| Version | AI score | Result |
|---|---:|---|
| v1_first10 | 0.941477 | AI (false positive under the supplied label) |
| multi_raw | 0.894517 | AI (false positive under the supplied label) |
| multi_no_rms | 0.706235 | AI (false positive under the supplied label) |
| multi_rms_normalized | 0.987317 | AI (false positive under the supplied label) |

## Volume sensitivity check

The decoded Dea waveform was multiplied by 0.1 (-20 dB amplitude), without editing or re-encoding the original file.

| Version | Original | 10 times quieter |
|---|---:|---:|
| v1_first10 | 0.941476727 | 0.938314485 |
| multi_raw | 0.894517330 | 0.871296869 |
| multi_no_rms | 0.706234617 | 0.706234545 |
| multi_rms_normalized | 0.987316723 | 0.987318817 |

Dropping RMS or normalizing it stabilizes the response to this gain change. Neither fixes authorship classification. The initial low-RMS contribution was an explanation of one model term, not proof that low volume caused the overall error: changing volume also changes RMS variation, and the net score depends on all terms.

## Verification and reproducibility

- The generalized extractor exactly matches v1 on the same ten-second samples.
- Section boundaries were checked on 30- and 120-second signals.
- A synthetic gain check initially used overly tight relative tolerances for near-zero frequency variations. The observed discrepancy was below 0.002 Hz. Comparing in the trained scaler units gave a maximum difference of 0.00000268, below the 0.0001 bound.
- Exported JSON predictions match the fitted sklearn pipelines to 1e-12.
- All 80 source audio hashes and section boundaries are saved in sections.json. Per-variant predictions.csv files contain splits, labels, scores and feature vectors.
- All source files, model hashes and feature/weight contributions for the external diagnostic are recorded in the JSON artifacts. The external MP3 remains outside the repository.

Train into a new directory (existing experiments are never overwritten):

```powershell
.\.venv\Scripts\python.exe baseline_v2.py train --output data/baseline_v2_rerun
```

Try the volume-feature-free version:

```powershell
.\.venv\Scripts\python.exe baseline_v2.py predict "C:\Users\38349\Downloads\TDS - Dea (Official Video HD).mp3" --model data/baseline_v2/multi_no_rms/model.json
```

## What remains

Streamlit still displays v1 and accurately labels its first-ten-second behavior. We have not silently replaced it with a model that still fails this diagnostic. The experiment implements the longer-audio alternative and volume comparisons, but has not established a reliable fix. Broader representative training data or stronger features remain the next substantive options; increasing the threshold to make this song pass is not an established solution.
