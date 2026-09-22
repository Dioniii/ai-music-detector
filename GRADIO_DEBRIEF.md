# Gradio demo debrief

Built on 2026-09-22 under the user's request to visualize the existing model before
working on Hugging Face encoders. The trained model and its preprocessing remain
unchanged. Gradio 6.28.0 and its dependencies are installed and recorded in uv.lock.

## What the demo does

`app.py` builds three tabs:

- **Analyze audio:** file upload, original-audio playback, prediction and raw score,
  waveform/spectrogram of the exact analyzed clip, feature contributions and a
  table of the ten measured and standardized values.
- **Model results:** saved holdout counts, confusion matrices for all three
  partitions, score distributions and the actual holdout errors.
- **How it works:** plain-language explanation of the existing pipeline and its
  limitations, including the absence of calibration, an inconclusive outcome and
  microphone validation.

Three locally available examples illustrate a correct human prediction, a correct
AI prediction and a known human false positive. Example labels come from the
existing saved predictions; examples are not evidence of generalization.

## How the backend connects

The upload component is `gr.File`, preserving the original encoded file for the
existing loader. Playback is a separate output so input audio is not re-encoded by
an audio input widget before prediction. `analyze` verifies the model's source-code
hashes and calls the same loader, preprocessing, feature extractor and saved-score
function as the command-line baseline. It neither retrains nor changes the model.

The exact ten-second mono clip is used both for features and signal plots. The
spectrogram is a visualization; the model still receives ten numeric summaries.
Raw audio file playback covers the original recording, labeled accordingly.

Feature influence is calculated as:

```
standardized = (feature - training_mean) / training_scale
contribution = standardized * learned_weight
logit = sum(contributions) + intercept
AI_score = sigmoid(logit)
```

The chart includes the intercept and sorts bars by signed contribution. Coral
points toward AI; green points toward human. Contributions are exact additive
terms in logit units, not percentage points, SHAP values, causal explanations or
proof of AI generation. The displayed score stays between zero and one without
being presented as a calibrated confidence percentage.

## Handling and local operation

Files must contain at least ten seconds. This demo also limits duration to five
minutes, upload size to 50 MB and decoded frame/channel count to 60 million sample
values, since the current loader holds the full recording in memory. Errors clear
previous plots and predictions; choosing another input also clears stale results.

Run `.\.venv\Scripts\python.exe app.py` from the repository root, then visit
http://127.0.0.1:7860. The server binds to localhost with share=False and analytics
disabled. Gradio temporary files and task server logs/screenshots are gitignored.
There was no public hosting, model publication, encoder download or remote audio
inference. Uploaded files are processed by the local Python server.

## Verification actually performed

- Built the UI with the installed Gradio version.
- Compared three real-file app predictions against the existing CLI backend.
- Verified contribution sums plus intercept reproduce its score to 1e-12 tolerance.
- Checked that missing input and the user's 3.96-second recording produce readable
  messages and clear previous outputs.
- Used the live Gradio client to upload a real recording and receive its score,
  both serialized plots and ten feature rows; also checked the short-file error
  through the live endpoint.
- Captured the page in a headless Edge browser. Found and corrected the dark-mode
  heading contrast. Visually inspected the rendered feature-contribution chart.
- Confirmed dependency-lock consistency and whitespace checks. No additional
  standalone diagnostic stage or model training was introduced.

The built-in browser tool failed because of the Windows sandbox helper. The
fallback used a hidden headless browser for rendering and the local Gradio client
for end-to-end upload/prediction. Browser mouse interaction was not automated.

Official Gradio references consulted: [Audio](https://www.gradio.app/docs/gradio/audio)
and [Blocks](https://www.gradio.app/docs/gradio/blocks). Installed API signatures
were inspected directly to match this version, including launch-time theme/CSS.

## What to try

Open the demo, choose the **known false positive** example, and press Analyze.
Inspect the contribution chart to see what this particular model relied on. Then
visit Model results to put that prediction in the context of its measured errors.
No claim is made that those influences identify synthetic production artifacts.

Prepared with Codex assistance. Hugging Face encoder comparison remains deferred.
