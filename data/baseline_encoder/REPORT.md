# EfficientAT encoder experiment - 2026-09-23

The encoder classifier is now the default Streamlit and CLI model. The frozen
EfficientAT mn10_as encoder supplies 960 learned features; our own logistic
regression was trained on 700 labeled recordings. No encoder fine-tuning,
threshold search or additional dataset collection was performed.

## Results on the same recording splits

| Model | Validation accuracy | Evaluation accuracy | Evaluation human false positives | Evaluation AI detected |
|---|---:|---:|---:|---:|
| Previous 80-recording demo | 68.0% | 61.3% | 25/75 | 42/75 |
| 1,000-recording handcrafted experiment | 63.3% | 63.3% | 21/75 | 41/75 |
| EfficientAT + our classifier | 88.7% | 86.7% | 11/75 | 66/75 |

The default was changed because validation improved. The evaluation recordings
have already been inspected in earlier experiments; these are reused benchmark
results, not a fresh blind test. The encoder classifier fits all 700 training
recordings correctly, which does not imply perfect generalization. Human false
positives remain 14.7% on this evaluation set.

Dea, a previously inspected user recording excluded from training, is classified
as human with AI score 0.015666. This score is not a calibrated probability.

## What changed

- features.py exports and loads the frozen encoder, and averages section embeddings.
- preprocessing.py supports a target sample rate while preserving the old 24 kHz
  default. EfficientAT resamples original decoded audio directly to 32 kHz.
- baseline.py trains either the historical features or encoder embeddings. Only
  the 700 training vectors fit StandardScaler and LogisticRegression (C=1,
  balanced weights, max_iter=1000, random_state=42, decision threshold=0.5).
- showcase.py and app.py display the encoder result, sampled audio and the ten
  strongest embedding contributions, plus a combined bar for the remaining 950.
  Dimensions are not presented as physical measurements. The three-tab layout,
  dark monospace styling, upload limits and serialized analysis are retained.

Selection stays at 500 human references from FMA Small and 500 generated Echoes
TTA recordings, including 12 generators. Related reference artists stay in one
split; previously inspected artists remain training-only. The splits are
700/150/150, with equal class counts in each. The sampler retains seed 42 and
up to five random 20-second sections, which may overlap.

Each section goes through the official evaluation mel frontend and convolutional
encoder, then frequency/time mean pooling. Section embeddings are averaged.
The old AudioSet classification head is omitted; the embedding dimensions do
not directly mean AI or human. Cached training embeddings stay local and are
keyed by decoded-audio hash, exported encoder hash and feature/preprocessing
source hashes. Scaler/weights remain portable numeric JSON.

## Verification and runtime

- All 121 existing tests passed, with one pre-existing short-spectrogram warning.
- Exported encoder matched official model embeddings at 10 and 20 seconds.
- The standalone export ran without torchvision or torchaudio installed.
- Repeated real-recording embeddings were identical; 960 finite values were returned.
- Saved classifier scores matched a sklearn refit and all 1,000 saved predictions.
- Streamlit AppTest exercised all three tabs, selected an example and produced
  the prediction, playback, waveform/spectrogram and contributions.
- The MP3 upload path matched the CLI result for Dea.

Local Windows CPU extraction for Dea (230.95 seconds, five sections): first run
3.16 seconds, warm run 0.67 seconds, peak process working set 574 MiB. These
numbers exclude the full Streamlit UI and do not establish cloud capacity.
The exported encoder is 11.6 MiB. CPU-only PyTorch is pinned in the lock file;
TorchScript remains usable in that version but is deprecated upstream. Future
runtime upgrades should recheck the export. License and exact source/checkpoint
hashes are in ../encoder/. No paid inference API or public deployment was used.

Known data limitations from the preceding experiment remain: one training MP3
emits a recoverable decoder warning; source labels and artist links are dataset
assumptions; evaluation covers Rock/Electronic but no Pop; generator, recording
length and encoding differences may be shortcuts. Microphone robustness and
unseen-generator performance have not been established.

## Commands

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -m streamlit run app.py
.\.venv\Scripts\python.exe baseline.py predict "path\to\recording.mp3"
.\.venv\Scripts\python.exe baseline.py train --encoder --output data/encoder_rerun
```

The prior demo remains available with
`--model data/baseline_random/demo/model.json`. Encoder features are saved in
features.npz with candidate IDs; metrics, predictions, sampled sections and
comparison are alongside this report. There are still six root Python modules.
