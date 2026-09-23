> Historical snapshot. For the current setup and file layout, see the [project README](../../README.md). Earlier raw experiment outputs are preserved in [the archive](../../data/archive/early_experiments.zip), using their original paths. Audio and plot links may refer to local-only files.

# First audio batch: selection, download and quality checks

Approved increment: 20 artist groups, one human reference and up to three generated
counterparts per group; at most 80 recordings and 200 MiB additional response-body
traffic. No model training, split assignment, publishing or package installation.

## Why this step exists

The candidate CSV describes available recordings. A model needs usable local
audio. This step connects those two stages while keeping dataset labels, audio
integrity and final eligibility separate. A successful download or decode does
not certify authorship, settle usage terms or establish detection performance.

## Deterministic selection

`batch_audio.py select` reads the existing candidate CSV. It sets aside references
with unresolved metadata flags and the known Straw Fields offset finding. From
references with a human row and all three requested generators, it orders artist
groups by SHA-256 of `batch-v1:` plus the group ID and takes the first 20 groups.
Within each group it chooses the lowest numeric reference ID, then the first
archive member in lexical order for each of ACE-Step, AudioLDM and MusicGen.
This order is fixed from metadata, without audio/model-score selection.

The result is 20 human references and 60 generated recordings. The chosen generator
strings are manifest labels; no held-out generator experiment is assigned here.
Their broad reference coverage permits the same three-generator composition for
every group. This narrow generator selection limits later generalization claims.

Echoes genre labels cover 12 Electronic, seven Rock and one Pop reference, giving
36/21/3 generated rows respectively. These are reference-linked Echoes genres,
not independently assigned FMA genre labels. The batch is deliberately small and
genre-skewed, not representative of all music or the eventual coffee-shop setting.

The Rolemusic reference is Scape from the city (91622), rather than the inspected
Straw Fields. Its four records still inherit `development_only=True`. Other
selected groups remain unsplit, rather than automatically becoming test data.
The selected batch does not contain any of the original six pilot files, although
receipt-checked reuse is supported and tested.

## Files and data flow

- `batch_audio.py`: select, download and inspect commands.
- `tests/test_batch_audio.py`: deterministic selection, flags, payload limit,
  receipts, no-network resume, plan identity, audio checks and PCM duplicate tests.
- `data/batch_manifest.csv`: selected candidate fields plus local file paths.
- `data/audio_batch/`: ignored original files and SHA-256 receipts. Original
  extensions are preserved; hashed candidate IDs provide stable local filenames.
- `data/source_metadata/batch_audio_transfers.json`: ignored cumulative transfer ledger.
- Its `.plan.json` companion binds the download to the selected plan and fixed cap.
- `data/batch_audio_results.json`: per-record check results and exact duplicate groups.

## Download behavior

The indexed compressed payload estimate is 84,473,438 bytes (80.56 MiB).
Selection refuses payloads above 180 MiB, leaving headroom below the 200 MiB
runtime cap. Actual range traffic also includes ZIP indexes, headers and repeated
reads; protocol/network overhead is not included in the response-body ledger.

The downloader uses the existing pinned Echoes URL and official FMA-small URL,
opens remote ZIPs through byte ranges, and reads only selected members. It rejects
full-body responses when servers ignore Range. Existing TransferBudget behavior
counts successfully read body bytes even for a truncated request and persists them.
Reruns retain the same cumulative cap; the ledger must not be deleted to reset it.

Each member must match indexed compressed/uncompressed sizes and the existing
8 MiB member ceiling. ZIP reading checks CRC before writing. Existing files are
reused only when their source/member, size and SHA-256 match their receipts.
Missing or conflicting receipts stop the operation rather than overwriting audio.
A changed batch plan is rejected. A local hash verifies later integrity; it is
not an independently published authorship certificate or source checksum.

## Audio checks and interpretation

The loader decodes float32 audio shaped `(frames, channels)` and rejects empty or
nonfinite audio. Whole-file checks record duration, sample rate, shape, peak,
RMS and each channel's mean. Files shorter than ten seconds, all-zero signals,
unsupported channel counts, peaks at/above full scale and absolute channel means
above 0.05 are flagged. The latter two are review triggers, not automatic exclusions,
clipping diagnoses or AI signatures. No normalization, repair or deletion occurs.

File SHA-256 groups byte-identical recordings. A second hash includes sample rate,
shape and decoded little-endian float32 samples; it catches identical decoded PCM
in different containers. Different encodings can produce slightly different
samples, so this does not replace near-duplicate or alternate-version review.
These checks cover this batch; they do not establish independence from every
unselected recording in the larger source datasets.

Listening/content confirmation, usage review and near-duplicate/alias review remain
pending. Candidate eligibility and split fields are not automatically rewritten.
Microphone/playback evaluation remains deferred as agreed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe batch_audio.py select
.\.venv\Scripts\python.exe batch_audio.py download
.\.venv\Scripts\python.exe batch_audio.py inspect
```

Selection and inspection outputs are created exclusively and refuse overwrites.
Run `select` only when creating a new approved batch manifest. `download` supports
receipt-checked resume; a complete batch needs no network. Do not regenerate or
change a batch/ledger to evade the approved cap. A missing download appears as an
integrity error if inspection is run before transfers finish.

## Verification

Tests were written before implementation and failed on the missing `batch_audio`
module. The completed suite passed 112 tests with the pre-existing short-audio
spectrogram warning. An independent manifest check verified 80 unique selected
rows, 20 groups, one reference and the same three generators per group, retained
provenance flags and unassigned splits. Completed download and real-audio outcomes are recorded below. The same inspection
functions ran on receipt-complete files as downloads progressed; the final results
were written only after all 80 records had been checked.

## Learning exercise

Find the four rows for reference 91622 in the batch manifest. Explain why all four
remain development-only even though this particular song was not in the pilot.
Then compare an encoded file hash with the decoded PCM hash: what kind of duplicate
can the second detect that the first might miss?

Prepared with Codex assistance under the approved bounded-download increment.

## Completed run - 2026-09-21

| Measurement | Result |
|---|---:|
| Downloaded recordings with receipts | 80 |
| Human / AI dataset labels | 20 / 60 |
| Successful finite-audio decodes | 80 |
| Files with at least ten seconds of audio | 80 |
| Exact file / exact PCM duplicate groups | 0 / 0 |
| Recordings with at least one quality flag | 30 |
| Channel-offset flags | 3 human |
| At/above-full-scale flags | 8 human, 20 MusicGen |
| Actual response-body transfer | 85,755,222 bytes (81.78 MiB) |
| Additional transfer cap | 209,715,200 bytes (200 MiB) |
| Recorded range requests | 250 |
| Pilot files reused in this particular selection | 0 |

One human recording has both flags, so the individual flag counts must not be
summed as distinct recordings. Peaks at/above full scale are not by themselves
proof of audible clipping. All originals remain unchanged. No zero-signal or
short-duration flag occurred; this does not establish that every segment is music.
Listening/content confirmation is pending for all 80 files.

The independent final artifact check confirmed the manifest hash, all 80 result
identities, receipt count, finite/duration checks and cumulative transfer total.
It also confirmed the empty exact-duplicate groups. There was no training, split
assignment, package installation, external upload or commit.

### Actual audio differences to track before modeling

| Source label | Sample rates | Channels | Duration range, seconds |
|---|---|---|---|
| Human references | 19 at 44.1 kHz, 1 at 48 kHz | 19 stereo, 1 mono | 29.977-30.003 |
| ACE-Step | 20 at 48 kHz | 20 stereo | 33.623-97.988 |
| AudioLDM | 20 at 16 kHz | 20 mono | 30.722 |
| MusicGen | 20 at 32 kHz | 20 mono | 29.940 |

These collection differences could influence a classifier. Converting everything
to 24 kHz mono does not restore bandwidth absent from the 16 kHz AudioLDM source,
erase codec history or remove an offset. No new preprocessing policy is applied
in this increment. The later experiment needs to account for these confounds.

The three human offset flags are:

- Scape from the city - Rolemusic: channel means -0.437876, -0.437876.
- I - Nocturnal Mayhem: channel means 0.060480, 0.023890.
- Only Instrumental - Broke For Free: channel means 0.064337, 0.062458.

### Concrete input-to-output trace

Human reference 114396, Breathe New Life by Scott Holmes, maps to
`fma_small/114/114396.mp3`. The batch stores it under the SHA-256-derived candidate
filename listed in the manifest, then checks its receipt and decodes it into a
float32 array of shape `(1321967, 2)` at 44,100 Hz. The two columns are channels;
1,321,967 frames divided by 44,100 gives approximately 29.977 seconds. Its measured
RMS is approximately 0.078405 and no automatic quality flag fired. This is a usable
decode result, not a detector score or final eligibility decision.

### Source and output hashes

- `data/candidate_manifest.csv`: `87b6bb9e9aa8ada4d7dfba8951663f70ab58c3a4d5851a1ccdf7aa843b1ffe09`
- `data/batch_manifest.csv`: `e7d09092fba135e61a03ad9315a7ab6f83147e62c30de00420dd0a22396dfb8a`
- `data/batch_audio_results.json`: `a09d9633628fa62b4c5c9e5c26abfb71188ad3535172469ff1e39746dc830d69`

## Where this leaves the project

The local audio batch and its integrity/quality evidence now exist. Listening,
usage and near-duplicate/alias review remain pending. The next discussion can use
these actual findings to decide a consistent quality policy before splitting and
training. The 20 artist-ID groups remain provisional; no claim of model accuracy
or coffee-shop readiness is supported by this increment.
