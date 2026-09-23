> Historical snapshot. For the current setup and file layout, see the [project README](../../README.md). Earlier raw experiment outputs are preserved in [the archive](../../data/archive/early_experiments.zip), using their original paths. Audio and plot links may refer to local-only files.

# Candidate manifest build

Date: 2026-09-21. Approved scope: expand local metadata into a recording-level
candidate CSV, with behavioral tests. No downloads, dependencies or training.

## What changed and why

[candidate_manifest.py](../../tools/candidate_manifest.py) joins the existing review to
Echoes rows and the two saved audio archive indexes. It produces one row per
recording in [candidate_manifest.csv](../../data/preparation/candidate_manifest.csv). Candidate labels
follow the approved historical-FMA/Echoes-TTA research standard, not certified
authorship or a numerical probability.

The function `build_candidates` receives five lists of dictionaries: human review,
Echoes manifest, FMA-small archive index, Echoes archive index, and pilot manifest.
It returns recording dictionaries in deterministic reference-ID/member-path order.
This step handles metadata only; no audio arrays or tensors are created.

Human members are located using the FMA ID: 112315 becomes
`fma_small/112/112315.mp3`. Each generated path must join to exactly one TTA row
with the correct reference and generator, and an indexed archive member. Invalid
joins, duplicate candidates, missing members and invalid sizes stop the build.
Unrelated Echoes rows are outside this candidate selection.

The archive URL and member identify a future download. `file_bytes` is the
uncompressed ZIP-member size (still an encoded audio file), and `compressed_bytes`
is the ZIP payload size. Neither is decoded waveform memory. Payload totals omit
ZIP indexes, range-read duplication and HTTP overhead, and include the six pilot
files already downloaded. They are not a future transfer budget.

## Measured output

| Measurement | Count |
|---|---:|
| Candidate recording rows | 1,458 |
| Human reference labels | 116 |
| Generated labels | 1,342 |
| Provisional artist-ID groups | 46 |
| Development-only rows | 164 |
| Rows inheriting a reference metadata-review flag | 110 |
| Compressed member bytes | 3,161,129,585 (about 2.94 GiB) |

The 110 flagged rows include seven human references and 103 generated counterparts.
Keeping them visible does not admit them to training. All rows have
`eligibility_status=needs_review` and `split=unassigned`. The 109 metadata-complete
references are not silently promoted to eligible audio.

## Grouping and evidence

`group_id` currently equals the reference artist group. Generated recordings inherit
that relationship; it does not mean the human artist produced the generated audio.
`grouping_status=provisional_artist_id` explicitly leaves cross-artist duplicates
and aliases unresolved. The CSV is not a final connected-component split.

For Digital Lightning, the review expands into one human row and 16 generated
rows. All 17 carry reference `fma:112315`, artist group `fma_artist:11018` and
`development_only=True`. Other songs by a pilot artist inherit that restriction
even if none of their audio was downloaded. The restriction is not a training split.

Human reference source, license and provenance fields remain attached to both
classes. AI rows separately use Echoes source and dataset license fields; they do
not inherit the FMA audio license as their own license. This records existing
metadata evidence and does not resolve pending usage review.

Every existing reference review reason is preserved verbatim. The existing
Straw Fields offset finding is explicitly retained on its human row as
`pilot_waveform_offset_review_pending`; this is a known finding, not an automatic
quality threshold or a flag applied to its generated counterparts. Integrity,
content/listening, usage and duplicate/alias review remain pending. Independent
authorship verification remains `unverified` without excluding records solely on
that basis under the approved research-label standard.

CSV booleans serialize as `True`/`False`: readers must parse these strings explicitly
rather than call `bool(value)`, which would treat even `False` as true.

## Reproduce and verify

Using the existing Python 3.12 environment, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe candidate_manifest.py --output data/candidate_manifest_rerun.csv
```

The CLI reads local files only and prints counts and input SHA-256 hashes. The
output directory must exist. Existing outputs are refused to preserve review work;
use a fresh filename for a rerun. No downloader is called.

Behavioral tests were written first and failed with the expected missing-module
error. After implementation, 17 new tests passed; the full suite passed with
**103 tests**, retaining the existing short-audio spectrogram warning. Tests cover
label/source separation, pilot restriction propagation, flag preservation,
determinism, input nonmutation, invalid joins/indexes and overwrite refusal.

An independent artifact check verified unique IDs/member paths, per-reference
counts and exact path sets, all unsplit/review statuses, preserved source flags,
the pilot restriction total and the Digital Lightning trace. No model accuracy
was measured.

## Reproducibility hashes

- `data/pilot_review.csv`: `f129437a5fa47f8618053f9648d53d08d03e959dac1ba58ea21be3bbf1ca02cd`
- `data/audio_pilot_manifest.csv`: `e555d5e00bb8d071c65e5f517bc98f3a689fd3daf681e2b03f938d756d1e48f7`
- `data/source_metadata/echoes_dataset_manifest.csv`: `d105d8b796419b7d1dd4317526db5b7fd88706e2d7aeafaabaf000128da46d1f`
- `data/source_metadata/fma_small_archive_index.json`: `cfd3f8d4a90d6c2e938b0383673a836cff48503c5085d1f92db7e7ef52ef9bcf`
- `data/source_metadata/echoes_archive_index.json`: `e90c66ceab1e29e403cf8285b0c1fcabc64be2a3dc2614cfaf4e5506f7be1b86`
- `data/candidate_manifest.csv`: `87b6bb9e9aa8ada4d7dfba8951663f70ab58c3a4d5851a1ccdf7aa843b1ffe09`

## Learning check

Why does a second song by a pilot artist remain development-only even if we have
never listened to it? Why does `split=unassigned` differ from `development_only`?

Exercise: read the CSV with `csv.DictReader` and count human and AI rows where
`development_only == "True"`. Expect 13 human and 151 AI rows. Do not assign splits
or edit review statuses yet.

Prepared with Codex assistance. Next work requires its own approved increment.
