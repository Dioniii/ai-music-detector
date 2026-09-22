# Audio-quality diagnostic debrief

Date: 2026-09-22. This approved experiment compares existing first-ten-second
features before and after channel-mean removal for all 80 recordings. Production
preprocessing, source audio, eligibility labels and splits are unchanged.

## What we built and why

[quality_diagnostics.py](../../quality_diagnostics.py) reuses the existing loader,
receipt checks, preprocessing and feature extractor. It adds a temporary mean-removal
operation and detailed full-scale counts. Nine behavioral tests cover per-channel
centering, nonmutation, valid shapes, whole-recording scope, finite input checks,
count denominators and constant-offset feature behavior.

The outputs are [features.csv](features.csv) (160 rows: two variants per recording),
[measurements.json](measurements.json) (80 source-audio measurements), and
[listening_checklist.csv](listening_checklist.csv) (13 pending review entries).
No new packages, downloads, models or training were involved.

## Exact data flow

1. Verify the local original against its source/member receipt and SHA-256.
2. Decode float32 samples shaped `(frames, channels)`.
3. Baseline: use unchanged `preprocess_audio(audio, start_seconds=0.0)`.
4. Experimental variant: compute each channel's whole-recording mean in float64,
   subtract it from that channel, cast to a new float32 array, then use the same
   preprocessing function at start zero.
5. Both variants become `(240000,)` mono clips at 24 kHz, each producing ten
   feature values in exactly `FEATURE_NAMES` order.
6. Count full-scale values on the original whole recording and its original first
   ten seconds, before resampling, channel mixing or mean removal.
7. Check the original hash again. No processed audio files are written.

The subtraction broadcasts a `(channels,)` mean across all frames. It removes a
constant channel mean; it is not a general high-pass filter, clipping repair or
loudness normalization. It can change zero crossings and spectral summaries as
well as RMS. Because the mean comes from the whole recording, centering does not
necessarily lower RMS in its first ten seconds. An inference input containing only
ten seconds would provide a different mean estimate. This experiment does not
choose between whole-recording, clip-only or filter-based production policies.

## Measured feature changes

| Human reference | RMS mean, baseline | RMS mean, centered | Centroid mean, baseline -> centered (Hz) |
|---|---:|---:|---:|
| Scape from the city - Rolemusic | 0.508414 | 0.258094 | 1934.03 -> 2424.48 |
| Only Instrumental - Broke For Free | 0.343479 | 0.330703 | 2653.58 -> 2701.78 |
| I - Nocturnal Mayhem | 0.152640 | 0.146685 | 1478.77 -> 1544.83 |

For Rolemusic, zero-crossing-rate mean also changes from 0.019653 to 0.064470.
The operation changes measured properties of this same recording substantially;
it does not reveal its authorship. This is direct evidence that the existing
channel offset matters to our feature representation.

| Source label | Median signed RMS change | Maximum absolute RMS change |
|---|---:|---:|
| human | 3.7241377e-11 | 0.25031979 |
| acestep | 4.9073479e-06 | 0.00018162148 |
| audioldm | -2.1311026e-06 | 3.1419774e-05 |
| musicgen | 5.8121197e-08 | 0.0025271102 |

Each source label has 20 recordings. These summaries are diagnostic measurements,
not classifier performance or an independent comparison of generation methods.
The paired CSV retains all ten features, not only the examples highlighted here.

## What the full-scale counts mean

`at_or_above_count` counts scalar decoded samples where absolute amplitude is at
least 1.0. `above_count` uses strictly greater than 1.0. The fraction denominator
is frames times channels. Per-channel fractions divide by frame count; the separate
frame count records whether either channel reaches the boundary in that frame.
These denominators prevent confusing stereo sample counts with time-frame counts.

| Source label | Whole-recording count range at/above 1 | Fraction range, percent |
|---|---:|---:|
| human | 0-2854 | 0.000000-0.107851 |
| acestep | 0-0 | 0.000000-0.000000 |
| audioldm | 0-0 | 0.000000-0.000000 |
| musicgen | 123-22120 | 0.012838-2.308784 |

All MusicGen samples counted here are exactly at the boundary, with none strictly
above it. This is compatible with a boundary-limited signal but does not establish
how it was produced or whether distortion is audible. We did not measure sustained
flat runs, listen to these files or infer clipping from the count alone.

Example: the MusicGen Breathe New Life recording has 958,080 mono samples at
32 kHz. Of those, 656 reach the boundary (0.068470%); within its first ten seconds,
19 of 320,000 samples do (0.005938%). A whole-file peak flag cannot express this
variation in frequency or location. Source encodings also affect decoded amplitude
behavior, so these counts must not be presented as generator fingerprints.

## Listening checklist

The 13 entries cover all ten distinct flagged human recordings plus one example
from each generator, selected by lowest reference ID. The generator examples all
reference The Factory by Multifaros; this keeps the reference consistent across
those three examples. The checklist paths point to unchanged original audio.

Listen to the original first ten seconds and extend to the whole recording where
needed to check content. Record music versus silence/corruption, obvious audible
distortion, and uncertain observations in `notes`; only then change that entry's
listening status. Do not label audio human/AI by ear or claim that a channel offset
must itself be audible. This checklist is a priority review set, not clearance of
the other 67 recordings. All listening statuses are currently pending.

## Verification and reproduction

The initial tests failed because the diagnostic module did not exist. After
implementation, all nine new tests passed. The full suite passed **121 tests**,
with the existing short-audio spectrogram warning.

The real run completed all 80 pairs. Independent artifact checks verified 160
finite feature vectors, paired variant identities, 80 original hashes matching
the prior inspection, and complete checklist coverage of flagged human files and
all three selected generators. Original hashes were also checked before and after
each comparison. No model metric was computed.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe quality_diagnostics.py --output-dir data/quality_diagnostics_rerun
```

Use a new output directory: the command refuses to overwrite existing outputs or
listening notes. The measurements JSON records source-manifest/inspection hashes,
feature order, centering scope and clip start. Existing feature/preprocessing
settings and dependencies remain in their modules and `uv.lock`.

## Evaluation consequence and decisions still pending

These whole-batch feature comparisons are now preprocessing-development evidence.
If we use them to choose a policy, this batch cannot later be described as an
untouched final test of that policy. Reserve fresh groups for such a final test;
no manifest splits or restrictions were silently rewritten here.

The experiment supports reviewing constant-offset handling, but does not establish
that mean removal improves detection. A decision still needs a production-compatible
scope and a consistent rule for both classes. Full-scale counts alone do not justify
excluding every flagged file. Listening, near-duplicate review and the previously
recorded source-bandwidth confound remain unresolved. Microphone testing stays
deferred. No production preprocessing decision was adopted in this increment.

## Learning check and exercise

Why can subtracting a constant change zero crossings and spectral centroid? Why
might a whole-recording mean differ from the mean of the submitted ten seconds?

Exercise: use the saved measurements JSON to calculate the first-ten-second
full-scale percentage for the MusicGen Breathe New Life example. Divide 19 by
320,000 and multiply by 100; expect 0.0059375%. Compare that with its whole-file
percentage without changing the audio or production preprocessing.

Prepared with Codex assistance under the approved diagnostic scope.

## Implementation and dependency hashes

- `quality_diagnostics.py`: `959cfad88aa53da63527f0464088a3ebb02ef28778d1495004e4b44578c32cb1`
- `preprocessing.py`: `5c183ff12942887d9b80174a80b5953ba7229c5adab17aff64cc1c53f04c4325`
- `features.py`: `065618508e38fbedf34f85d15e7ac85e1f2a82c0a3c8e34db1f1bc6bde9f635d`
- `uv.lock`: `55d4df1ea78f7c7098505c687d5c19b4cc0762003771521e2f763512ee9adc10`
