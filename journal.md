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

## Increment 6: five audio features, ten summaries - 2026-09-19

- Added `features.py` with `extract_features(clip)` and an explicit `FEATURE_NAMES`
  order. Input is a preprocessed `(240000,)` floating array at 24 kHz; output is
  a `(10,)` float64 vector. Wrong shapes, integer PCM, NaN and infinity are rejected.
  Arrays do not contain sample-rate metadata, so callers must use preprocessing.
- Installed librosa 0.11.0 and locked its dependencies. The required dependencies
  include scikit-learn, numba and llvmlite; no classifier was used or trained.
  No models or audio were downloaded. Existing source files and input clips were
  hash-checked before/after extraction and remained unchanged.
- Measured the same first ten seconds of all six pilot recordings, explicitly
  using `start_seconds=0.0`. Saved [data/pilot_features.csv](data/pilot_features.csv)
  and [data/pilot_features_config.json](data/pilot_features_config.json). The latter
  records versions, settings, feature order and source/output hashes.
- The resulting numeric matrix is `(6, 10)`. Identity, label expectation, reference
  group, generator, start time, source hash and review note are metadata columns,
  not additional model inputs. Use FEATURE_NAMES to select the ten measurements.

### Definitions and data flow

Each clip is analyzed using left-aligned, non-padded 1024-sample frames and a
512-sample hop: a 42.667 ms window every 21.333 ms. There are 467 complete frames;
the last 384 samples (16 ms) do not form another full frame and are omitted.

| Feature | Measurement | Unit |
|---|---|---|
| RMS | Root mean square of sample amplitudes in each frame | Relative amplitude |
| Zero-crossing rate | Fraction of sign changes per frame; exact zero is treated as positive | Dimensionless |
| Spectral centroid | Magnitude-weighted center of the spectrum | Hz |
| Spectral bandwidth | Magnitude-weighted spread about the centroid, p=2 | Hz |
| Spectral flatness | Geometric/arithmetic mean ratio of power bins, using a numerical floor | Dimensionless |

RMS and zero crossings use the unwindowed frame. Spectral features use a Hann
window and a shared magnitude spectrogram shaped `(513, 467)`: 513 frequency bins
and 467 frames. The five frame-feature arrays stack to `(5, 467)`. Taking each
row's mean and population standard deviation (`ddof=0`) produces `(5, 2)`, then
flattening yields `(10,)` in FEATURE_NAMES order. Standard deviation describes
variation over time, not uncertainty about authorship or classifier confidence.

Feature order is RMS mean/std, ZCR mean/std, centroid mean/std, bandwidth mean/std,
then flatness mean/std. RMS is not perceptual loudness. Centroid and bandwidth
use magnitude weights; flatness uses power with floor `1e-10`.
The [librosa 0.11 feature documentation/source](https://librosa.org/doc/0.11.0/_modules/librosa/feature/spectral.html)
documents these calculations.

Silence gives zero RMS, ZCR, centroid and bandwidth but flatness **1**, because all
power bins are floored equally. That convention must not be described as noise
evidence. Very low-energy frames can also be affected by this floor.

### Actual pilot measurements (selected columns)

| Sample | Mean RMS | RMS std | Mean centroid (Hz) |
|---|---:|---:|---:|
| human_45101 | 0.604507 | 0.100834 | 1542.015 |
| human_127294 | 0.432164 | 0.039296 | 2664.063 |
| human_112315 | 0.275001 | 0.060915 | 2376.946 |
| ai_45101 | 0.048570 | 0.013985 | 599.422 |
| ai_127294 | 0.039010 | 0.011941 | 977.396 |
| ai_112315 | 0.101173 | 0.062838 | 3366.064 |

Example: Digital Lightning goes through the existing preprocessing into 240,000
samples. Its 467 frame RMS values summarize to mean 0.275001 and std 0.060915;
those become positions 0 and 1 in the feature vector. Its centroid mean is about
2376.946 Hz, position 4. These are measured properties, not forensic findings.

All three human-labeled clips happen to have higher mean RMS than the three AI
clips. This six-record sample does not establish a useful classification rule:
mastering, source preparation, clip position and the known offset are confounds.
Straw Fields retains an explicit review note; its offset contributes energy and
affects zero crossings/spectral measurements. No recording was repaired, relabeled,
or excluded silently. Listening and provenance review remain pending.

### Verification and learning review

- New tests first failed because `features.py` did not exist. Twelve feature tests
  then passed: output contract/order, unchanged input, silence, known tone energy,
  frequency/zero crossings, noise versus tone, varying energy, constant offset,
  invalid shape/type and non-finite inputs.
- Full suite: **86 passed**, with the existing one-segment audio plotting warning.
  All six actual pilot vectors contain ten finite values. No accuracy was measured.
- Mean/std discard temporal order and many musical details; their usefulness must
  be evaluated later. No scaler was fitted, and no training/test split was created.

To inspect a vector in the project Python interpreter:

```python
from audio_inspection import load_audio
from preprocessing import preprocess_audio
from features import extract_features, FEATURE_NAMES

audio = load_audio("data/audio_pilot/human_112315.mp3")
clip = preprocess_audio(audio, start_seconds=0.0)
vector = extract_features(clip)
print(dict(zip(FEATURE_NAMES, vector)))
```

- Understanding checks: why do five features become ten numbers? Why is higher
  RMS in these three human examples insufficient to claim an AI-detection rule?
- Exercise: extract features from `clip * 0.5`. Predict which measurements will
  change most, then compare. RMS mean/std should approximately halve; normalized
  spectral measurements should remain similar except where the flatness floor
  matters. Do not modify the original audio files.
- Pause for review; dataset expansion, splitting and training require a new step.


## Increment 7: evaluation protocol - 2026-09-21

- Created [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md), version 0.1, under
  the approved documentation-only scope. No records were relabeled or split.
- Separated metadata completeness from eligibility: provenance, listening and
  quality questions remain unresolved and must be reviewed before modeling.
- Defined transitive reference/artist/duplicate groups and reserved the six
  inspected pilot tracks and their connected groups for development. Split
  proportions remain pending an inventory of eligible groups.
- Defined separate seen-generator and unseen-generator evaluation conditions,
  training-only fitting, validation-based threshold selection, track-level
  metrics and explicit handling of inconclusive predictions.
- Recorded pending decisions, experiment freeze records and future behavioral
  checks. No code, dependencies, audio downloads or training were added.
- Verified protocol local links and basic Markdown structure. The previous
  86-test result is historical; tests were not rerun for this documentation step.
- Learning exercise: sketch linked songs and artists, then explain why a held-out
  generator output linked to a training reference cannot be a clean final test.


## Dataset eligibility inventory - 2026-09-21

- Reviewed existing manifests and recorded [eligibility/group counts](data/eligibility_inventory.md).
- Metadata-complete candidates: 109 human references, 45 artist-ID groups and 1,239 associated TTA paths. Outside the three pilot artists: 96 references, 42 groups and 1,088 paths. Groups remain provisional pending duplicate/alias review.
- Confirmed that the pilot development restriction covers 13 reference tracks and 151 generated paths through shared artists.
- Deferred microphone validation as requested; direct-file modeling remains the first target.
- All 116 provenance flags remain unverified. A proposed dataset-label research scope needs an explicit decision before admission; no labels or splits were changed.
- Checked manifest ID/path integrity and count consistency. No production code, packages, downloads or training; unit tests were not rerun.


## Research-label standard approved - 2026-09-21

- The user approved historical FMA human reference labels and Echoes TTA generated labels for an exploratory baseline after discussing authorship uncertainty.
- Updated evaluation protocol to version 0.2 and the inventory decision section. Independent authorship certification is no longer an admission requirement; unverified provenance flags remain an accurate evidence record, not an automatic exclusion.
- No numerical certainty is assigned to dataset labels. Model scores are separately subject to evaluation and calibration; they do not certify authorship.
- Per-record source, usage, audio quality/content and duplicate checks remain required. Direct-file modeling comes first; microphone testing stays deferred.
- Documentation-only change; no manifest relabeling, production code, downloads, split assignment or training. Checked local links and whitespace; unit tests were not rerun.


## Candidate manifest implementation - 2026-09-21

- Built a local-only candidate manifest builder and 17 behavioral tests under the approved scope. Tests first failed because the module did not exist.
- Generated 1,458 candidate rows: 116 human reference labels and 1,342 generated labels across 46 provisional artist groups. The pilot restriction propagates to 164 rows (13 human, 151 generated).
- Preserved source evidence, license separation, unresolved reference flags and the existing Straw Fields offset finding. All rows remain needs_review and unassigned; no authorship verification flags changed.
- Indexed compressed member payloads total 3,161,129,585 bytes, about 2.94 GiB; this is not a download budget and includes existing pilot members.
- Full suite: 103 passed, with the existing short-audio spectrogram warning. Independent artifact checks confirmed counts, path joins, restrictions and flags.
- See [candidate manifest report](data/candidate_manifest_report.md) for the concrete Digital Lightning trace, reproduction commands, source hashes and learning exercise.
- No dependencies, audio downloads, training, commits or split assignments. Pause for review before the next increment.


## First bounded audio batch - 2026-09-21

- Implemented `batch_audio.py` and nine new behavioral tests under the approved 20-group, at-most-80-file, 200 MiB additional-transfer scope. Initial tests failed on the missing module; final full suite: 112 passed with the existing short-audio spectrogram warning.
- Selected one reference per group using a fixed metadata-only order, with one ACE-Step, AudioLDM and MusicGen counterpart each. Saved 80 manifest rows with existing flags and unassigned splits; the Rolemusic group stays development-only.
- Downloaded all 80 selected members with size/CRC checks and SHA-256 receipts. Actual additional response-body transfer: 85,755,222 bytes (81.78 MiB), within 200 MiB. No pilot file happened to overlap this batch; receipt-checked no-network resume is tested.
- Checked completed files as they arrived. All 80 decode to finite audio of at least ten seconds. No exact-file or exact-PCM duplicate groups were found within the batch; near-duplicate review remains pending.
- Found quality flags on 30 recordings: three human channel offsets and 28 full-scale flags (eight human, all 20 MusicGen), with one overlap. Preserved originals; flags are review triggers, not AI evidence or automatic exclusions.
- Source rate/channel/duration differences remain confounds, including AudioLDM at 16 kHz. Listening/content and usage review remain pending; no eligibility promotion, model training or split assignment occurred.
- See [batch report](data/batch_audio_report.md) for measured results, the Scott Holmes array-shape trace, reproducibility instructions and exercise. Audio/receipts stay ignored by git; no packages, uploads or commits.


## Paired audio-quality diagnostic - 2026-09-22

- Added `quality_diagnostics.py` and nine behavioral tests. Initial tests failed on the missing module; final full suite: 121 passed with the existing short-audio spectrogram warning.
- Compared existing first-ten-second features with and without whole-recording per-channel mean removal for all 80 files. Saved 160 feature rows, 80 full-scale measurements and a 13-entry listening checklist covering ten flagged humans plus three generator examples.
- Rolemusic's Scape from the city RMS mean changed from 0.508414 to 0.258094, showing that the offset strongly affects the representation. This is not a detection result or a production-policy adoption.
- MusicGen boundary-hit fractions range from 0.012838% to 2.308784% across whole recordings; none of those hits exceeds the boundary. Counts alone do not establish audible clipping or generation provenance.
- Receipt and before/after hash checks confirmed unchanged originals. No downloads, packages, training, split changes or edits to production preprocessing. Listening remains pending.
- These whole-batch comparisons are preprocessing-development evidence; a policy chosen from them needs fresh groups for an untouched final test. No split restrictions were automatically rewritten.
- See [diagnostic debrief](data/quality_diagnostics/REPORT.md) for data flow, measured tables, reproducibility, limitations and the denominator exercise.


## First trained exploratory baseline - 2026-09-22

- Following the user's direction to prioritize the portfolio baseline over additional diagnostic stages, implemented training and local-file prediction in `baseline.py`. No new test suite or diagnostic stage was added.
- Kept original preprocessing and the ten existing baseline features. Split the 20 provisional artist groups deterministically into 12 training, four validation and four exploratory holdout groups, with the known pilot group forced into training.
- Fit training-only StandardScaler plus balanced logistic regression (C=1, fixed threshold 0.5). Saved portable numeric model parameters, all 80 predictions/assignments, metrics and a detailed report; added a concise project README.
- Holdout: 11/12 AI examples detected, 2/4 human false positives; AI precision 84.6%, recall 91.7%, human FPR 50%, accuracy 81.25%. Validation human FPR is 75%. These are exploratory results from a previously inspected batch, not an untouched final or unseen-generator test.
- Actual execution verified group separation, convergence, finite features, exported score agreement and a fresh-file feature/prediction round trip. Existing unit tests were not rerun. Existing scikit-learn package was promoted to a direct dependency; offline lock validation passed after preserving locked package entries. No downloads or installations.
- Pending quality/source flags remain documented; no original audio, production preprocessing or source eligibility fields were altered. No commits or publication.
- See [baseline report](data/baseline_v1/REPORT.md) for actual errors and reproduction commands. We now have a working trained backend whose main measured weakness is human false positives.


## Local Gradio showcase - 2026-09-22

- Added `app.py` and installed/locked Gradio 6.28.0. Built audio-analysis, model-results and pipeline-explanation tabs around the existing saved baseline.
- Visuals show the exact analyzed waveform/spectrogram, ten feature values, exact additive logit contributions including intercept, raw score, confusion matrices, score distributions and recorded errors. Local examples include a known human false positive.
- Verified three real predictions against the CLI and contribution reconstruction to 1e-12. Live Gradio uploads returned the expected score, plots and feature table; missing/short files clear outputs and show readable errors.
- A headless browser screenshot exposed a dark-mode heading contrast issue, which was fixed. Visually checked the contribution chart. Built-in browser access was unavailable due to the sandbox helper; browser interaction was not automated.
- Kept original audio preprocessing and model unchanged. No new diagnostic stage, model training, public deployment or encoder download. The app runs on localhost with sharing and analytics disabled.
- Updated README and [Gradio debrief](GRADIO_DEBRIEF.md). Temporary uploaded audio and server logs/screenshots are gitignored. Hugging Face encoders remain deferred.


## 2026-09-22 - Streamlit migration

Replaced Gradio with Streamlit for a zero-budget Community Cloud deployment path.
Preserved the three tabs, dark monospace design, example placement, model and
processing functions. Extracted shared analysis/figures into showcase.py and
authored copy into showcase_content.py. Added per-session results, isolated upload
temporary files, focused UI checks, theme configuration, and deployment notes.
All three example scores match the unchanged CLI. No retraining, source-audio
upload, commit, or public deployment was performed. See STREAMLIT_DEBRIEF.md.


## 2026-09-22 - Mobile layout finish

Reduced mobile hero and upload spacing, added 44-52 px touch targets, wrapped long
file names, and kept charts/tables readable inside horizontal scroll regions.
Desktop charts retain their full-width presentation. CSS file changes now invalidate
the cached stylesheet. Browser checks at 320 and 390 px found no page-width overflow;
the Analyze button is fully visible within an 844 px viewport at both widths.
On a 390 px screen, evaluation charts retain 760 px content within a 358 px scroll
region. Model code and predictions are unchanged. No deployment performed.


## Longer-audio / volume experiment

Implemented features_v2.py and baseline_v2.py: up to five 20-second sections,
mean feature aggregation, and three volume-treatment variants. Trained on the
unchanged 48/16/16 recording splits. Validation selected multi_raw; it did not
improve the old holdout. RMS normalization reduced old-holdout human false
positives from 2/4 to 1/4 but also reduced AI detection from 11/12 to 10/12,
and did not improve validation. All variants still classify Dea as AI. Volume
normalization and removing RMS stabilize scores under a 0.1 gain change, which
is distinct from correcting authorship classification. No deployed model or
threshold changed. Full results: data/baseline_v2/REPORT.md.


## Random-section sampling integrated

Added features_random.py and baseline_random.py without modifying the earlier
model contracts. Fixed-seed PCG64 selects one position from each legal-start
range; training and prediction use the same rule. Trained all three feature
variants; validation selected multi_raw. Connected that model to Streamlit through
showcase_random.py, updated timestamps/coverage plots/copy, and regenerated
evaluation artifacts and example labels. Cached evaluations now include the
artifact digest; switching model versions clears prior session output. Dea
remains AI at 0.797715. The new old-holdout result is 2/4 human false positives
and 10/12 AI detections. No accuracy improvement is claimed.
See data/baseline_random/REPORT.md for method, checks and commands.


## Simplified beginner workflow

Consolidated the 17 root Python files into six: audio_inspection, preprocessing,
features, baseline, showcase, and app. Removed the six redundant variant/copy
modules and moved five optional dataset-preparation scripts into tools/.
The current trainer fits only the chosen raw-feature random-section model;
unused experiment branches no longer run during inference.

The saved experiment results remain historical records. Runtime checks now
validate the model feature order and sampling policy rather than requiring
retraining whenever a source-file comment changes. Original source hashes in
the current model are labeled training provenance; learned parameters are
unchanged. The existing tests were updated for the tools import paths.
README.md now explains the six-file flow and one train/predict command.
Future work should extend these files unless a separate module has a clear need.

Verification: all 121 existing tests pass. A temporary training run reproduced
the saved scaler and coefficients within 1e-10; its temporary output was removed.
Dea retained its score and exact sampled timestamps, and Streamlit example
analysis returned both charts and ten feature rows. The data-tool module entry
point and consolidated baseline command also launch successfully.


## Expanding to 1,000 recordings: selection and method

Approved scope: 500 human-reference recordings and 500 AI recordings. Extended
the existing tools/batch_audio.py instead of adding another Python module.
The selection is recorded in data/dataset_1000.csv, with source URLs, archive
members, labels, artist/reference IDs, license metadata, local paths and splits.
The companion dataset_1000_plan.json records the counts.

The human pool comes from FMA Small and the AI pool from Echoes TTA. Twelve
generators replace the earlier three-generator subset. Pop, Rock and Electronic
counts are matched between the classes within each split, to reduce an obvious
genre shortcut. Metadata selection uses a deterministic SHA-256 ordering; it
does not use model scores. Artists and related reference recordings stay in one
split. All previously inspected artist groups are restricted to training.

The fixed split is 700 training (350 per class), 150 validation (75 per class),
and 150 holdout (75 per class). The AI side has 29 reference-artist groups in
training, seven in validation and nine in holdout. More recordings therefore
do not mean equally many independent source groups. FMA artists are spread
more broadly. Labels and artist IDs remain dataset assumptions.

Selected compressed payload is 1,684,271,518 bytes, about 1.57 GiB. Download
only the selected ZIP members using byte ranges, reusing existing receipt-
verified files. Completed files have hashes and receipts, so interrupted
downloads can resume. Audio remains gitignored; no paid API is involved.
FMA selection includes CC BY, BY-SA, BY-NC and BY-NC-SA metadata, excludes ND,
and retains individual license URLs. This does not grant general redistribution
permission; the downloaded recordings are not bundled into the public app.

The planned training comparison keeps the same ten raw features, random
20-second sampling policy, scaler, logistic regression and 0.5 threshold.
Only the training split fits the scaler and classifier. Compare against the
saved 80-recording model on the same new validation and holdout recordings.
Use validation accuracy (balanced classes) to decide whether to switch the demo;
the holdout reports the result rather than selecting model settings.
No additional hyperparameter search, microphone augmentation or encoders are
part of this expansion. Exact decoded duplicates are rejected during extraction.
Near-duplicate and listening reviews are not being added to this portfolio step.

Manifest validation confirmed 350/350 training and 75/75 in both evaluation
splits, without artist or reference overlap. The selected holdout contains
Electronic (38 per class) and Rock (37 per class), with no Pop. Validation
contains all three genres. Holdout metrics therefore do not establish Pop
performance. This is reported as a limitation, not corrected after looking at
model outcomes.

During extraction, FMA track 29245, "The Angel - Benjamin Bret", emitted an
mpg123 dequantization warning. It decoded to 30.0027 seconds of finite samples
with nonzero RMS (0.11886), and was retained in training. No claim is made that
its source encoding is clean. Validation and holdout do not contain this track.
This is a recorded data-quality limitation rather than a silent exclusion or
an outcome-driven change to the fixed selection.

### Completed result

All 1,000 selected recordings downloaded and extracted successfully. Exact
decoded duplicates: zero. Model/artifacts are saved in data/baseline_1000/.
The same-new-holdout comparison is old 61.3% vs new 63.3% accuracy, 25/75 vs
21/75 human false positives, and 42/75 vs 41/75 AI detections. Validation accuracy
fell from 68.0% to 63.3%, with unchanged 25/75 human false positives and AI
detections falling from 52/75 to 45/75. Following the recorded validation rule,
the demo model remains unchanged. No threshold or hyperparameter search was
performed in response to these outcomes.

Dea changed from AI score 0.797715 to 0.458691 (human side) with the experimental
model. It remained outside training and selection. The detailed report records
this diagnostic, the mixed outcome and data limitations for the eventual article.
The next step should examine the representation before assuming more recordings
alone will solve the remaining errors. No new Python files were added.

Verification: all 121 existing tests passed (one pre-existing short-spectrogram
warning). The exported model reproduced all 1,000 saved scores within 1e-12;
its scaler mean and scale matched only the 700 training feature vectors. The
manifest hash matched, all three splits contained both classes and all 12
generators, and the previous demo model hash remained unchanged. The repository
still has six root Python modules. The new model JSON is 1,698 bytes.


## EfficientAT integrated - 2026-09-23

Added the frozen mn10_as encoder within features.py and extended the existing
trainer with --encoder. No new Python modules. Official source and checkpoint
hashes, attribution and a standalone 11.6 MiB export are recorded in data/encoder.
Audio is resampled directly to 32 kHz; the same seed and random-section policy
remain. The model averages 960-dimensional embeddings and trains our own scaler
and logistic regression on the same 700 training recordings. Embeddings are
cached locally for reuse.

Validation improved from the previous demo's 68.0% to 88.7%, so the encoder
classifier is now the default. Reused evaluation accuracy is 86.7%, with 11/75
human false positives and 66/75 AI detections. Dea is human-side at 0.015666;
it was not used for training or selection. No threshold search was performed.
The full method and limitations are in data/baseline_encoder/REPORT.md.

Streamlit retains the three tabs and styling, with correct encoder-specific
copy and contribution labels. All 121 existing tests pass. Encoder export
parity, deterministic extraction, saved-classifier parity and Streamlit
example/upload flows passed. Local extraction was 3.16 seconds cold, 0.67 warm,
574 MiB peak on Dea; hosted capacity remains unverified. No public deployment,
commit or push was performed. The earlier classifiers remain available.
