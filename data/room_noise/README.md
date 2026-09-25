# Background audio for the final room-augmentation experiment

Nine five-second ESC-50 clips (about 3.8 MiB total): rain, washing machine and
breathing, with separate source recordings for training, validation and evaluation.
These approximate environmental noise, not realistic cafe chatter or all devices.
The small selection is a deliberate portfolio-scale compromise.

The downloaded WAV files stay local and are not needed by the deployed app.
`manifest.json` records the exact upstream revision, URLs, original source IDs,
split assignments and SHA-256 checksums. `LICENSE.txt` retains upstream licensing
and per-source attribution. ESC-50 is distributed under CC BY-NC 3.0; its ESC-10
subset uses CC BY 3.0. This experiment is for the noncommercial portfolio project.

Source: https://github.com/karolpiczak/ESC-50
Citation: K. J. Piczak, ESC: Dataset for Environmental Sound Classification,
Proceedings of ACM Multimedia, 2015, DOI 10.1145/2733373.2806390.

From the repository root, restore the pinned files with:

```powershell
uv run --locked python -m tools.batch_audio download-room-noise
```

The training code filters the music, adds six short echoes, adjusts its volume,
and mixes a looping noise clip with a deterministic random offset. Original
noise files are unchanged. No microphone recording from the user is used for
training or evaluation.
