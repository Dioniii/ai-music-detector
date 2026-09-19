# Six-file audio pilot

Date: 2026-09-19. This is a download/decoding sanity check, not a detector experiment.

## Outcome

All six selected MP3s downloaded through HTTP byte ranges, passed ZIP CRC checks,
and decoded with the existing SoundFile installation. No new dependencies were
needed. No full archives, models, training, uploads, or publication were involved.

Pilot transfer (including ZIP indexes): **10,771,811 response-body bytes, 10.27 MiB**,
below the 32 MiB ceiling. Selected compressed audio totals 8,868,531 bytes; the six
local MP3s total 8,955,574 bytes. Transfer receipts exclude HTTP/protocol overhead.

The pilot uses three distinct human artist IDs and three generators. Genre labels
come from Echoes, not a new listening-based annotation. Human/AI labels are source
expectations, not conclusions made by our code. Human authorship is not independently
verified; all six listening checks remain pending.

## Files and measurements

| Local sample | Reference title | Source | Rate | Channels | Duration (s) |
|---|---|---|---:|---:|---:|
| [human_45101.mp3](audio_pilot/human_45101.mp3) | Straw Fields — Rolemusic | FMA small | 44,100 | 1 | 29.977 |
| [ai_45101.mp3](audio_pilot/ai_45101.mp3) | Straw Fields reference | Echoes / ACE-Step | 48,000 | 2 | 83.778 |
| [human_127294.mp3](audio_pilot/human_127294.mp3) | Autopsy — Oh Yeah, the Future | FMA small | 44,100 | 2 | 29.977 |
| [ai_127294.mp3](audio_pilot/ai_127294.mp3) | Autopsy reference | Echoes / Suno | 48,000 | 2 | 168.720 |
| [human_112315.mp3](audio_pilot/human_112315.mp3) | Digital Lightning — Cloudkicker | FMA small | 44,100 | 2 | 30.003 |
| [ai_112315.mp3](audio_pilot/ai_112315.mp3) | Digital Lightning reference | Echoes / Udio | 48,000 | 2 | 130.859 |

AI examples are text-generated counterparts, not synchronized covers or matching
excerpts. Each human excerpt is about 30 seconds; its generated counterpart is a
longer recording. None of the files were resampled, mixed, trimmed, normalized,
or otherwise altered after retrieval.

All six duration checks pass within a declared 0.25-second tolerance, and all
decoded samples are finite. None is entirely zero. These checks do not rule out
partial silence, unexpected speech, incorrect titles, or semantic mislabeling.

Full machine-readable measurements and SHA-256 hashes are in
[audio_pilot_results.json](audio_pilot_results.json). Selection, exact archive
members/sizes, source/license URLs, grouping IDs and pending listening status
are in [audio_pilot_manifest.csv](audio_pilot_manifest.csv).

## What the inspection revealed

1. **Sample-rate and duration confounds:** all three human examples are 44.1 kHz
   and about 30 seconds; all three generated examples are 48 kHz and longer.
   A rule using only those properties could separate this tiny collection without
   learning anything about AI generation. These are sample-specific observations,
   not universal properties of human or AI music.
2. **Channel differences:** Straw Fields is mono; the other five files are stereo.
   Channel handling must become consistent before a useful modeling experiment.
3. **Unusual offset in Straw Fields:** its mean sample amplitude is **-0.522756**,
   consistent with the visibly downward-shifted waveform. Retain the original
   and flag this recording for listening/source/decoder review before training.
   The cause has not been established; no repair or replacement was attempted.
4. **Amplitude differences:** measured RMS ranges from about 0.094 to 0.591.
   RMS is not a perceptual loudness measurement; the large offset in Straw Fields
   contributes to its RMS. Do not interpret these values as provenance evidence.
5. Some decoded peaks exceed magnitude 1 (Autopsy human, Digital Lightning human,
   and its Udio counterpart). Values were preserved and reported. This alone
   establishes neither audible clipping nor a particular cause.
6. Waveform/spectrogram images for all six files were visually inspected. Labels
   are readable and the plots show signal over time. Color scales are selected
   separately by the existing plotting function, so equal colors across images
   do not necessarily mean equal power. Only the first channel is plotted.

Plot links: [Straw Fields human](pilot_plots/human_45101.png),
[ACE-Step](pilot_plots/ai_45101.png), [Autopsy human](pilot_plots/human_127294.png),
[Suno](pilot_plots/ai_127294.png), [Digital Lightning human](pilot_plots/human_112315.png),
[Udio](pilot_plots/ai_112315.png).

## Provenance, storage, and reproducibility

- FMA-small archive: `https://os.unil.cloud.switch.ch/fma/fma_small.zip`.
  The selected human metadata records list CC BY 4.0; their titles, artists, and
  original track URLs are retained in the manifest for attribution and review.
- Echoes archive revision: `14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2`.
  The selected release is documented under CC BY-SA 4.0. These terms are recorded
  as source evidence; this step does not authorize redistribution or publication.
- Audio and per-file download receipts are ignored by git under `data/audio_pilot/`.
  Generated plot PNGs are ignored under `data/pilot_plots/`.
- The separate `data/source_metadata/pilot_audio_transfers.json` ledger enforces a
  cumulative 32 MiB body-byte cap for this audio pilot, including the FMA-small
  index preflight. Earlier metadata audits have their own existing ledger.
- Downloading is restricted to six fixed archive members. Changed member sizes
  are refused. Range-ignoring servers are refused by the existing transfer helper.
  Existing files are never overwritten; files with matching local receipts can
  be skipped after hash verification. A repeat download command still reads ZIP
  indexes and consumes a small amount of the remaining budget.

```powershell
# Already downloaded; normally you only need the local inspection command.
.\.venv\Scripts\python.exe pilot_audio.py inspect
.\.venv\Scripts\python.exe -m pytest -q

# Open one interactive plot window manually:
.\.venv\Scripts\python.exe audio_inspection.py data/audio_pilot/human_112315.mp3
```

`pilot_audio.py inspect` writes PNGs without opening a window and records decoding
errors in the JSON instead of silently dropping a file. It uses the shared loader
and plotting code. For example, Digital Lightning decodes to `(1323119, 2)` samples;
`1323119 / 44100 = 30.002698` seconds. Statistics use both channels, while plots
use channel 1. This step introduces no learned features or classifier.

## Tests and remaining work

Tests were written first and failed because `pilot_audio` did not exist. The
offset measurement test also failed before that measurement was implemented.
Final full suite: **46 passed**, with one pre-existing short-audio plotting warning.
Tests cover member selection, unexpected sizes, overwrite refusal, decoded shapes,
duration matching, silence reporting, corrupt audio, and unchanged files when
measuring channel offsets.

Manual listening remains pending. Listen to a human excerpt and its linked
generated counterpart, and note whether each contains music, silence, obvious
distortion, or unexpected content. That is a content check, not proof of authorship.
Further preprocessing, replacement files, dataset expansion, or training requires
another approved increment.
