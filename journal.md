# Engineering journal

## Increment 1: Audio inspection — 2026-09-19

### Built and decisions

- A local audio inspector prints sample rate, channel count, array shape, and duration, then displays a waveform and spectrogram of channel 1.
- SoundFile loads float32 samples with shape `(frames, channels)`, even for mono. No resampling, channel mixing, or volume normalization is applied.
- NumPy handles sample arrays; Matplotlib plots and computes the spectrogram. The transform uses 1024-sample Hann windows with 512-sample overlap. A longer window resolves nearby frequencies better but blurs changes over time.
- Spectrogram values are power density on a decibel scale, not measured acoustic sound pressure. A fixed floor keeps silence finite.
- Python 3.12.10 and uv 0.12.5 were verified. Dependencies are recorded exactly in `uv.lock`. Tests use pytest. `.gitignore` excludes the local environment and generated caches.
- AI assistance: implementation, synthetic tests, and this journal were prepared with Codex under the user's approved scope. No detector or performance claim exists yet.

### Verification and failures

- Before implementation, pytest failed during collection with `ModuleNotFoundError: No module named 'audio_inspection'`, the expected missing-module failure.
- After implementation: **11 passed**, with one Matplotlib warning about a single spectrogram segment for the one-sample fixture.
- Checks cover mono/stereo preservation, duration, missing/unreadable/empty/non-finite audio, tone frequency, silence, very short input, metadata output, and plot rendering.
- A generated one-second, 1000 Hz tone at 8000 Hz printed shape `(8000, 1)` and duration `1.000 s`. Its rendered PNG was visually inspected: labeled axes and a bright band near 1000 Hz. The waveform looks filled at this scale because it contains 1000 cycles; zooming reveals individual cycles.
- Sandbox access initially prevented running the existing Python interpreter. Running the approved commands outside that restriction worked; no new Python installation was needed. Default uv cache access also failed, so setup used `--no-cache`.

### Run locally (PowerShell)

```powershell
uv sync --locked
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe audio_inspection.py "C:\path\to\short-music.wav"
```

If the default uv cache is inaccessible, use `uv --no-cache sync --locked`.

### Limits and open questions

- No user-selected music file has been inspected; a real-file check and interactive plot-window behavior remain unverified.
- The whole file loads into memory. Use short WAV files for now. Other formats are not covered by these tests.
- Only channel 1 is plotted. All channels remain available in the loaded array.
- Audio shorter than one window is zero-padded for the spectrogram only. Its apparent spectral time span can exceed the recording's duration; padding adds no information.
- These tests verify code behavior. Neither the plots nor the synthetic tone evaluate AI detection or prove provenance.

### Learning review and next experiment

Explain why `(8000, 2)` at 8000 Hz lasts one second, and why channel 1 is selected with `samples[:, 0]`. Try changing the synthetic tone to 500 Hz and updating its expected peak. Next, inspect a short local WAV after the user selects it; further implementation requires discussion and explicit approval.
