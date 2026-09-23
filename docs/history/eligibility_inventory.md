> Historical snapshot. For the current setup and file layout, see the [project README](../../README.md). Earlier raw experiment outputs are preserved in [the archive](../../data/archive/early_experiments.zip), using their original paths. Audio and plot links may refer to local-only files.

# Dataset eligibility and group inventory

Review date: 2026-09-21. Local metadata review only; no audio downloads, label changes, split assignment or training.

## Scope

The first model targets direct audio files. Coffee-shop playback and microphone validation are deferred. Source metadata establishes research labels and grouping; users will not need to provide source information, and it is not a model feature.

## Counts reproduced from the existing manifests

| Candidate pool | Human references | Artist-ID groups | Associated TTA paths |
|---|---:|---:|---:|
| All provisional candidates | 116 | 46 | 1342 |
| Metadata complete | 109 | 45 | 1239 |
| Metadata complete, outside pilot artists | 96 | 42 | 1088 |

These are planning counts, not eligible or independently verified audio counts. Artist-ID groups are provisional: artist aliases, alternate recordings or audio duplicates could connect groups further. Unique paths do not prove unique audio. No final-test groups have been selected.

## Pilot exclusion propagates to related tracks

| Pilot artist | Candidate human references | Associated TTA paths |
|---|---:|---:|
| Rolemusic | 2 | 21 |
| Cloudkicker | 1 | 16 |
| Oh Yeah, the Future | 10 | 114 |

Together these three artist groups cover 13 human references and 151 TTA paths, rather than just the six downloaded files. Under the evaluation protocol, these groups remain development-only if later eligible. They are not necessarily all assigned to training.

## Generator coverage outside pilot artists

Counts below measure distinct candidate artist groups with at least one associated path, not verified audio or resolved generator families.

| Manifest generator | Candidate artist groups |
|---|---:|
| acestep | 42 |
| audioldm | 42 |
| brev | 26 |
| diffrhythm | 42 |
| elevenlabs | 26 |
| mubert | 26 |
| musicgen | 42 |
| producer | 26 |
| songgen | 42 |
| stableaudio | 27 |
| suno | 26 |
| udio | 26 |

Coverage suggests a grouped experiment may be feasible, but it does not establish enough final-test examples for a precise false-positive estimate. Generator aliases and audio eligibility remain unresolved; no held-out family or split percentages are selected.

## Eligibility review outcome

- All 116 human references remain `provenance_status=unverified`. Their metadata catalog-entry years range from 2009 to 2017. These dates are historical evidence, not recording dates or independently verified production histories.
- Seven Katapulto references have the previously flagged legacy public-domain URL; they remain set aside pending review. This review makes no new legal determination.
- The other 109 references pass existing metadata-completeness checks, not authorship certification. The larger audio pool still needs decode, content and duplicate checks; pilot listening remains pending.
- Straw Fields retains its unresolved waveform-offset quality flag. No repair or exclusion was applied.
- Echoes generation labels and reference links are dataset-supplied evidence. This inventory does not independently reproduce generation or verify every audio member.

## Approved label standard - 2026-09-21

After discussing the limits of provenance evidence, the user explicitly approved
an exploratory dataset-label scope: historical FMA recordings as human references
and Echoes TTA recordings as generated examples. We evaluate agreement with these
labels, acknowledging possible label errors; we do not claim certified authorship
or assign an invented percentage of label certainty.

Independent authorship certification is no longer required for this exploratory
baseline. Existing provenance flags remain unverified as an accurate evidence
record. Source consistency, usage terms, audio integrity/content, quality and
leakage checks still apply. No candidate has been automatically marked eligible,
and no splits, downloads or training are authorized by this label decision.

## Verification and reproducibility

Read `pilot_review.csv` with Python csv.DictReader. Count rows for references, distinct artist_id values for provisional artist groups, and summed tta_count for associated paths. Filter metadata_status == metadata_complete, then exclude artist IDs occurring in audio_pilot_manifest.csv for the last row above. Generator coverage counts distinct artist IDs whose generators_json contains the named generator.

Checked unique reference IDs, nonempty artist IDs matching raw_artist_id, globally unique associated paths and agreement between parsed path totals and tta_count. No production code was changed and the unit suite was not rerun.

Input SHA-256 hashes:

- `data/pilot_review.csv`: `f129437a5fa47f8618053f9648d53d08d03e959dac1ba58ea21be3bbf1ca02cd`
- `data/audio_pilot_manifest.csv`: `e555d5e00bb8d071c65e5f517bc98f3a689fd3daf681e2b03f938d756d1e48f7`

Related documents: [audit](audit.md), [evaluation protocol](EVALUATION_PROTOCOL.md).

Learning exercise: explain why the ten Oh Yeah, the Future references must stay together even though only one was used in our pilot.

Prepared with Codex assistance.
