# Streamlit Community Cloud deployment

The app is prepared for deployment; it has not been published by this migration.
Use **Streamlit Community Cloud**, the free service at https://share.streamlit.io,
with this existing CPU model. No Snowflake trial, paid API, credit purchase,
GPU service, or separate backend is needed.

## Publish when ready

1. Commit and push the project changes to your GitHub repository.
2. Sign in to Community Cloud with GitHub and choose **Create app**.
3. Select the repository and branch; set the entrypoint to **app.py**.
4. In Advanced settings choose **Python 3.12**. No secrets are required.
5. Deploy, then open the resulting `streamlit.app` link and try an audio upload.

Do not select a paid product or enter payment details for this deployment.
Free hosting has resource and availability limits. If limits are reached, accept
an unavailable demo instead of moving to a paid service. Provider policies can
change; this project does not promise permanent hosting availability.

## Files the deployment uses

- The six main Python files listed in README.md.
- `data/baseline_encoder/model.json`, `metrics.json`, `predictions.csv`.
- `data/encoder/efficientat_mn10.pt`, `provenance.json`, and `LICENSE.txt`.
- `assets/streamlit.css` and `assets/fonts/`, including the font license.
- `.streamlit/config.toml`, `.python-version`, `pyproject.toml`, `uv.lock`.

Training and dataset-download scripts are not run during deployment. The optional
`tools/` scripts and original audio files are not required for inference.

Community Cloud recognizes `uv.lock` and installs the locked Python environment.
Do not add a competing requirements file. The lock includes the audio dependencies
and Streamlit; Gradio is removed. SoundFile's supported Linux wheel bundles
libsndfile, so this app does not require a separate FFmpeg process.

The training audio directories remain gitignored. The example selector appears
only when example files are present. A cloud deployment from this repository
therefore supports uploads, results and explanations without downloading the
training dataset. Publishing example audio would be a separate redistribution
choice, with its source attribution and permissions reviewed first.

## Runtime behavior

Uploads are limited to 50 MB, 10 seconds to 5 minutes, and 60 million decoded
sample values. The original preprocessing still decodes/resamples the full file
before selecting up to five reproducibly random 20-second sections. Concurrent analyses are serialized to
bound simultaneous decoding and protect Matplotlib. Per-session results are not
shared between visitors. Uploaded temporary files are removed immediately after
analysis; the playback bytes remain in that session while the result is shown.

The Linux cloud build and real hosted memory limits still need verification on
the first deployment. Local UI checks do not establish hosted capacity.

Official references (checked 2026-09-22):
- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies


## EfficientAT runtime

The default classifier now uses the frozen EfficientAT encoder. `uv sync --locked`
installs pinned CPU-only PyTorch; export-only torchvision/torchaudio are in the
optional encoder-build dependency group and are not needed by the app. The
11.6 MiB encoder artifact is local, so inference does not download weights or
call a paid service. Training embeddings and downloaded source stay gitignored.

The encoder is loaded once per process, uses two PyTorch threads, and processes
sections sequentially. App analysis remains serialized. The local extraction
benchmark was about 574 MiB peak working set; this excludes the full UI and is
not a guarantee of hosted memory use. The app is verified locally; actual
Community Cloud deployment and resource capacity remain unverified. Keep the
free-only policy above.


## Phone microphone recordings

Select **Use microphone**, allow browser microphone access, record 20-30 seconds
of music, stop, then select **Analyze recording**. Recordings must be at least
10 seconds and at most 5 minutes. The native Streamlit recorder requests 32 kHz
WAV audio and uses the same size limits, temporary-file cleanup and prediction
pipeline as uploads. No new package or paid service is required. Switching input
methods clears the displayed result so a previous upload is not analyzed by mistake.

Phones need a trusted HTTPS app URL and microphone permission. Opening the
laptop's plain HTTP LAN address on a phone is not sufficient. Localhost works for
testing on the computer running the app; it does not refer to that computer when
opened on a phone. If access is blocked, check site permissions or upload a file
recorded with the phone's recording app. Do not disable browser security to test.

Adding recording support does not establish accuracy on phone microphones or
music in noisy rooms. Real iOS/Android capture and hosted HTTPS permissions still
need a device check after deployment.

References: [Streamlit audio input](https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input)
and [browser microphone requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).
