# Frozen EfficientAT encoder

This folder contains the locally runnable encoder, its MIT attribution, source
provenance and local benchmark. The app makes no model API calls and does not
need the training dataset or the upstream source checkout.

`efficientat_mn10.pt` contains the official evaluation mel frontend and frozen
mn10_as convolutional features, with frequency/time mean pooling. The original
AudioSet classification head is omitted. The output is 960 numbers per section;
our own logistic regression learns the human/AI decision from those numbers.
The exported artifact is approximately 11.6 MiB.

Training and prediction use mono 32 kHz audio, up to five randomly positioned
20-second sections, seed 42, and the mean of section embeddings. Native audio
is resampled directly to 32 kHz, rather than upsampling the old 24 kHz features.
There is no augmentation or gradient computation at prediction time.

The upstream source revision and checkpoint/export SHA-256 hashes are recorded
in provenance.json. The export matched official outputs at 10 and 20 seconds.
TorchScript is supported by the pinned runtime but deprecated upstream; keep the
lock file when deploying. A future runtime upgrade should recheck or re-export
this artifact rather than silently replacing libraries.

Normal setup: `uv sync --locked`. Only CPU PyTorch is needed for inference.
To reproduce the export in a fresh checkout without an existing .pt artifact:

```powershell
uv sync --locked --group encoder-build
.\.venv\Scripts\python.exe features.py export-encoder
uv sync --locked
```

Export refuses to overwrite an existing artifact. Build-only source/checkpoints
are downloaded to ignored data/encoder_source. Dataset embedding caches live
in ignored data/embedding_cache. No new Python modules were added.

The local Windows benchmark for a 230.95-second recording with five sections
was 3.16 seconds on the first extraction, 0.67 seconds warm and 574 MiB peak
working set. Repeated embeddings were identical. This measures a local extraction
process, not the full Streamlit app or hosted capacity. Cloud deployment has
not been performed.

The runtime narrowly suppresses the known torch.jit.load and legacy STFT
deprecation notices only for pinned PyTorch 2.14.0, around the relevant calls.
This does not migrate those APIs. Other warnings remain visible, and upgrading
PyTorch disables these filters. Real-recording prediction parity was verified.
