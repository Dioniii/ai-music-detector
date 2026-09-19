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

## Increment 2: Echoes/FMA metadata audit - 2026-09-19

- Approved scope: source metadata only, a small audit module and tests, and a report;
  maximum transfer 400 MiB. No audio downloads or training.
- Implemented `dataset_audit.py` using Python's standard library, retaining source
  CSVs and SHA-256 receipts locally under ignored `data/source_metadata/`.
- HTTP range reads retrieved only ZIP indexes and selected CSVs: 21,000,502
  response-body bytes (20.03 MiB). No new dependencies were needed.
- Found 119 unambiguous FMA-small references linked to 1,389 Echoes TTA rows.
  After setting aside three NoDerivatives references and conflicting paths, the
  provisional pool is 116 references and 1,342 TTA rows. This is not a verified
  or balanced training set.
- Found 16 ambiguous reference names, repeated MusicGen paths, a 296-versus-300
  reference-count discrepancy, and per-track license terms needing review.
- Initial missing-module failure was observed before implementation. Two further
  regression tests failed before adding real-data anomaly diagnostics. Full suite:
  26 passed, one pre-existing short-audio warning.
- Full evidence, source links, exact counts, limitations and reproduction commands:
  [data/audit.md](data/audit.md).
- Learning check: Why must a duplicate title/artist pair stay ambiguous? Why are
  1,342 generated rows not equivalent to 1,342 independent human references?
- Exercise: add a second matching FMA record to the exact-match test fixture and
  predict the resulting status before running it.
- Next experiment to discuss after review: resolve source IDs/license links for
  a small pilot and define reference/artist grouping before proposing audio downloads.

## Increment 3: pilot review manifest - 2026-09-19

- Added the approved metadata review to `dataset_audit.py` and wrote
  [data/pilot_review.csv](data/pilot_review.csv): 116 candidate reference rows,
  source/license URLs, original dates, artist/reference grouping keys and linked
  Echoes paths. No packages, audio, training or publication.
- Original FMA metadata retrieval consumed 7.91 MiB against an enforced 15 MiB
  incremental ceiling. Total archive metadata traffic is now 27.93 MiB.
- 109 candidates have complete metadata; seven retain a review flag for a legacy
  public-domain URL. All 116 remain unverified for human authorship. The 109 rows
  link 1,239 TTA paths; all 116 together link 1,342 unique paths from 46 artists.
- Most recording dates are absent (111 of 116); catalog dates are retained in
  their own field and are never substituted for recording dates.
- Initial tests failed for the missing function. Regression tests caught an
  inaccurate legacy-license diagnosis and a missed explicit version conflict.
  Final full suite: 39 passed, one existing audio warning. Artifact counts and
  unique paths were independently checked after generation.
- The seven historical public-domain URLs are recognized as an older CC tool;
  they are flagged for review, not silently rewritten as CC0. The report links
  the official explanation and records the raw metadata hash.
- Learning check: Why are metadata completeness and verified authorship separate?
  Why would splitting by recording alone still permit artist overlap?
- Exercise: remove `license_url` from the complete raw-record fixture by setting
  it to an empty string, predict the status/reason, then run the pilot-review test.
- Remaining: manual provenance review and decisions about pilot size and grouped
  splits. Further implementation and audio downloads await a new approved step.

## Increment 4: six-file audio sanity check - 2026-09-19

- Downloaded three FMA-small excerpts and three linked Echoes text-generated files
  (ACE-Step, Suno, Udio) using selective ZIP reads, exact member-size checks, ZIP
  CRC validation, and SHA-256 receipts. Audio and plots remain local and git-ignored.
- Added `pilot_audio.py`, behavioral tests, the six-file manifest, measured JSON
  results, and [data/audio_pilot_report.md](data/audio_pilot_report.md).
- Pilot traffic was 10.27 MiB under a 32 MiB cap, including archive-index reads.
  Existing SoundFile decoded all six MP3s. No new dependencies or training.
- All files match expected durations within 0.25 seconds, have finite samples,
  and are not entirely zero. All six rendered plots were visually inspected.
- Learned why preprocessing matters: these human files are 44.1 kHz/~30 s;
  generated files are 48 kHz/~84-169 s. Those properties could become shortcuts.
- Straw Fields has mean amplitude -0.522756 and a visibly shifted waveform.
  Its cause is unresolved; preserve it and flag it for review rather than silently
  repairing it. No files were normalized, resampled, trimmed, or replaced.
- Initial missing-module failure and a failing offset-measurement test were
  observed before implementation. Final full suite: 46 passed, one existing warning.
- Listening and content identity checks remain pending; source labels remain
  expectations. Six records cannot support detector accuracy claims.
- Learning check: why would a 44.1-vs-48-kHz rule be misleading? For a stereo
  array shaped `(1323119, 2)`, which dimension determines duration?
- Exercise: use `audio_inspection.py` to open Digital Lightning's human plot, then
  its Udio counterpart; compare time axes and listen to a short section of each.
- Pause for review before proposing preprocessing or additional downloads.

## Increment 5: consistent ten-second clips - 2026-09-19

- Added `preprocessing.py`: `preprocess_audio(audio, start_seconds=...)` accepts
  the existing decoded `Audio` object and returns a new one-dimensional float32
  array of exactly 240,000 samples (10 seconds, mono, 24,000 Hz).
- Channels are averaged, then the full recording is resampled, then the requested
  clip is extracted. The start is required and rounds down to the 24 kHz sample
  grid, by less than 1/24000 second. Requested intervals beyond the original
  duration are rejected before resampling; no short recording is padded to pass.
- SciPy 1.18.1 was installed (35.0 MiB reported package download) and recorded in
  `uv.lock`. No existing package changed. The package declaration is in
  `pyproject.toml`; no dataset/model downloads or training occurred.
- [SciPy's resample_poly documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
  describes the low-pass filtered resampling used here. Parameters are explicit:
  Kaiser window with beta 5.0, constant zero boundary extension. The ratio for
  44,100 to 24,000 Hz is 80/147; for 48,000 to 24,000 Hz it is 1/2. Filtering
  reduces high-frequency aliasing rather than merely dropping samples or relabeling
  the sample rate. Content above the target Nyquist frequency of 12 kHz is lost;
  24 kHz is an initial baseline choice, not a promise about a future encoder.
- Tests were written first and failed for the missing module. All 28 new tests
  pass, including an 18 kHz input that would alias without filtering, a 1 kHz tone
  whose frequency/amplitude are preserved, mono/stereo handling, exact/fractional
  starts, short clips, invalid input, and unchanged arrays. Full suite: **74 passed**,
  with the existing one-segment plotting warning.
- Ran all six local pilot recordings with explicit `start_seconds=0.0` as a smoke
  check, not a proposed training clip-selection policy. Every output was finite,
  shape `(240000,)`, dtype `float32`, 24,000 Hz, and 10 seconds. SHA-256 checks
  confirmed both original files and input arrays were unchanged.
- Measured configuration, versions, source/output hashes and per-file statistics
  are saved in [data/preprocessing_pilot_results.json](data/preprocessing_pilot_results.json).
  Derived clips were held in memory, not written as new audio files.
- Concrete trace: Digital Lightning loads as `(1323119, 2)` at 44,100 Hz.
  Averaging axis 1 produces `(1323119,)`; resampling changes the sampling grid
  to 24,000 Hz while preserving playback speed. Starting at zero and taking
  240,000 samples yields ten seconds, with mean amplitude about 0.00000244.
- Limits: opposing stereo channels can cancel; clipping from the beginning may
  select silence/intros; resampling has boundary effects and does not erase codec
  history. This whole-recording implementation uses memory proportional to the
  recording's length. Dataset splitting remains separate and must precede future
  clip generation for training/evaluation.
- No volume normalization, peak clipping, or DC-offset removal was added. Straw
  Fields' first ten seconds retain mean amplitude about -0.546410. Its source
  review remains pending; formatting success does not make it a clean training
  example. We have not listened to or assessed the perceptual quality of the outputs.

### Try the function

Start the project Python interpreter with `.\.venv\Scripts\python.exe`, then:

```python
from audio_inspection import load_audio
from preprocessing import preprocess_audio, TARGET_SAMPLE_RATE

audio = load_audio("data/audio_pilot/human_112315.mp3")
clip = preprocess_audio(audio, start_seconds=0.0)
print(clip.shape, clip.dtype, len(clip) / TARGET_SAMPLE_RATE)
# Actual pilot contract: (240000,) float32 10.0
```

- Understanding checks: why does `mean(axis=1)` remove channels rather than time?
  Why would changing a sample-rate number without resampling change playback speed?
- Exercise: change `start_seconds` to 5.0. Predict the output shape and duration,
  then run it. Try 25.0 on this 30-second file and explain the error.
- Pause for review. No subsequent increment or training is authorized by this step.
