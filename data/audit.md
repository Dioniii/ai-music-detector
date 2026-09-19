# Echoes + FMA small: metadata audit

Date: 2026-09-19. This report describes actual metadata checks, not a trained detector.

## Decision

The combination supports a small exploratory experiment, subject to per-track
provenance/license review and an audio integrity check. It does not provide a
large, balanced, verified dataset automatically. The matched pool is small and
genre-skewed; do not use every generated row as an independent human counterpart.

No audio members were downloaded or decoded. No training, package installation,
external upload, git commit, or publication was performed in this increment.

## Sources and reproducibility

- Echoes archive: [pinned Hugging Face release](https://huggingface.co/datasets/Octavian97/Echoes/tree/14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2).
  Member: `Echoes/dataset_manifest.csv`.
  SHA-256: `d105d8b796419b7d1dd4317526db5b7fd88706e2d7aeafaabaf000128da46d1f`.
- FMA archive: [official metadata ZIP](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip).
  Member: `fma_metadata/tracks.csv`.
  SHA-256: `f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b`.
  The ZIP URL is not revisioned, so this extracted-member hash identifies the audited content.
- Echoes [revised paper](https://arxiv.org/html/2603.23667v2) and
  [dataset card](https://huggingface.co/datasets/Octavian97/Echoes) describe the
  current release under CC BY-SA; the card specifies version 4.0. The old paper's
  MIT statement was superseded in the revised paper.
- [FMA documentation](https://github.com/mdeff/fma) distinguishes CC BY 4.0 metadata
  from each audio track's own license. Dataset/code licenses do not replace audio licenses.

HTTP byte ranges fetched ZIP indexes and selected metadata members rather than
whole archives. Python's ZIP reader checked member CRCs. Actual response-body
transfer: **21,000,502 bytes (20.03 MiB)**, below the approved 400 MiB cap. This
ledger excludes transport headers/overhead. FMA's extracted CSV is 260,414,445
bytes on disk despite its much smaller compressed transfer size.

Local evidence in ignored `data/source_metadata/`:

- Original CSVs and `.receipt.json` files with member hashes and source URLs.
- `transfers.json`: requested ranges and response-body bytes.
- `echoes_archive_index.json` and `fma_archive_index.json`: archive entries and sizes.
- `reference_matches.csv`: all generated rows with match status, candidate FMA IDs,
  FMA-small membership, artist ID, and license text.
- `summary.json`: machine-readable counts reproduced by the report command.

## What was measured

| Measurement | Result |
|---|---:|
| FMA metadata records | 106,574 |
| Echoes manifest rows | 4,468 |
| Text-to-audio (TTA) rows | 3,165 |
| Audio-to-audio (ATA) rows | 1,303 |
| Distinct reference strings | 296 |
| Unambiguous exact title/artist references in all FMA | 280 |
| Ambiguous references in all FMA | 16 |
| Missing reference strings | 0 |
| Unambiguous references in FMA small | 119 |
| Unambiguous references outside FMA small | 161 |
| FMA artist IDs among the 119 matched small references | 49 |
| TTA rows linked to the 119 small references | 1,389 |
| Matched small references with NoDerivatives license text | 3 |
| Provisional references after exclusions below | 116 |
| Provisional TTA rows after exclusions below | 1,342 |

The provisional pool excludes references with NoDerivatives license text and
all rows whose audio path appears more than once. It is a count for planning,
NOT legal clearance, verified human authorship, or a guarantee of distinct audio.
Broad labels such as `Attribution` still need exact license links and attribution
details before selecting audio. NoDerivatives references are set aside conservatively;
this report does not claim that their licenses categorically forbid model training.

### TTA availability by generator

Generator names below preserve the actual manifest strings; `producer` should
not be silently renamed to the paper's Riffusion label without confirming the mapping.

| Manifest generator | All TTA rows | Provisional small-linked TTA rows |
|---|---:|---:|
| acestep | 294 | 115 |
| audioldm | 292 | 114 |
| brev | 298 | 136 |
| diffrhythm | 299 | 117 |
| elevenlabs | 300 | 136 |
| mubert | 149 | 66 |
| musicgen | 296 | 116 |
| producer | 151 | 69 |
| songgen | 292 | 115 |
| stableaudio | 194 | 86 |
| suno | 300 | 136 |
| udio | 300 | 136 |

Provisional generated-row genre counts: Electronic **429**, Pop **711**, Rock
**202**. These are Echoes' genre labels, not an assertion that FMA's genre taxonomy
matches them. The pool is imbalanced in both class count and genre.

## Matching rule and concrete examples

Construct `FMA track title + " - " + artist name` and look it up against
`original_audio`. Search all FMA records BEFORE checking subset membership, so
an ambiguous name is not resolved merely because one candidate is in small.
If exact lookup fails, try only Unicode NFC, case, and whitespace normalization;
report that as `normalized`, never `exact`. This release needed no normalized
matches. No fuzzy matching is used.

- `Acoustic Unleashed - Remain` matches FMA track **141875**, artist **22429**,
  subset `small`, license text `Attribution`. One associated AI row is
  `TTA/acestep/Acoustic_Unleashed_Remain_acestep_TTA_001.mp3`.
- `1984 - Punk Rock Opera` matches two FMA track IDs: **137212** and **149410**.
  It remains ambiguous; neither is selected automatically.

These are metadata links, not content fingerprints. Alternate versions, duplicate
audio, and incorrect source metadata remain possible.

## Defects and unresolved questions

1. The revised paper says 300 human references; the manifest contains **296 distinct
   reference strings**. Different recordings may share names, so this is not proof
   that four recordings are missing. The 16 ambiguous references need stable IDs.
2. Two paths each occur three times: `TTA/musicgen/_musicgen_TTA_001.wav` and
   `ATA/musicgen/_musicgen_ATA_001.wav`. That is **four extra rows** beyond unique
   path counts. All six affected rows require review; do not arbitrarily keep one.
3. Three matched FMA-small references have NoDerivatives license text, unlike the
   paper's stated CC0/CC-BY/public-domain source selection:
   `Different Day - First` (13197), `Ponky McFonky - My brother Daniel` (117288),
   and `Soul of the Demon - Zombie Raiders` (37147).
   This discrepancy may reflect versions or metadata history and remains unresolved.
4. All distinct manifest audio paths occur in the Echoes ZIP index. Its directory
   structure contains TTA/ATA audio and the CSV; no separate human-audio collection
   was found. Presence in an archive index does not verify decodability or content.
5. A historical FMA catalog and named artists are provenance evidence, not
   independent confirmation of every recording's production process. Music from
   a recent arbitrary FMA upload must not automatically inherit a human label.
6. FMA small contains 30-second excerpts while Echoes has varied lengths. We must
   use consistent clip duration, channel handling and sample rate later, and check
   codec/loudness/source confounds. Converting formats does not erase codec history.
7. Split original-reference groups before clipping, and group shared human artists
   where feasible. Reserve unseen generators only with a documented protocol that
   also controls reference/artist overlap. More generated rows do not create more
   independent human references.

## Reproduce

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe dataset_audit.py report data/source_metadata/echoes_dataset_manifest.csv data/source_metadata/fma_tracks.csv
```

For a fresh metadata directory, these commands fetch only indexes and named CSVs:

```powershell
.\.venv\Scripts\python.exe dataset_audit.py list echoes
.\.venv\Scripts\python.exe dataset_audit.py fetch echoes Echoes/dataset_manifest.csv
.\.venv\Scripts\python.exe dataset_audit.py list fma
.\.venv\Scripts\python.exe dataset_audit.py fetch fma fma_metadata/tracks.csv
```

Downloads are explicit, refuse full responses when range requests are ignored,
share a persistent 400 MiB body-byte budget, and refuse to overwrite existing
source CSVs. Local report regeneration performs no downloads. Do not delete the
transfer ledger to bypass its cumulative cap. Source metadata remains out of git.

## Verification

- Initial tests failed because `dataset_audit` did not exist.
- Two additional regression tests failed on missing summary fields before adding
  diagnostics for the defects found in the real metadata.
- Final full suite: **26 passed**, with the existing one-segment warning from the
  audio-inspection silence fixture. No new audit warnings.
- Tests cover exact, normalized, missing and ambiguous matches, FMA header parsing,
  TTA counts, duplicate paths, per-track license counts, budget persistence and
  range refusal, and reading a metadata member without reading a synthetic audio member.

AI assistance: Codex wrote the audit, tests and report under the approved scope.
Further implementation or audio downloads require a new approved increment.
