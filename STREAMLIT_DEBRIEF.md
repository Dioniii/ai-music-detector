# Gradio to Streamlit: implementation debrief

## Why we switched

The goal is a public portfolio demo with a zero hosting budget. Streamlit
Community Cloud hosts Streamlit apps for free. The earlier Gradio interface
worked, but the hosting options considered added account restrictions or limited
resources. Switching the interface lets us use Community Cloud while retaining
our own trained classifier. This migration does not retrain the model or improve
its measured accuracy.

## What stayed the same

The saved model, scaler, threshold, feature order, preprocessing, and training
code are unchanged. Audio is decoded, averaged to mono, resampled to 24 kHz,
then the first ten seconds produce ten audio features. The saved scaler and
logistic regression weights produce the same uncalibrated AI score.

The three tabs remain Analyze audio, Model results, and How it works. The same
waveform, spectrogram, contribution bars, confusion matrices, score distributions,
and actual holdout mistakes are shown. Shared plot and analysis functions moved
from the old app.py into showcase.py. The authored explanatory text is held in
showcase_content.py so it can be edited independently of widget code.

## How the new interface works

`app.py` is now a Streamlit script. Streamlit reruns it when a widget changes.
We use session state to keep a completed result visible across those reruns.
Analysis runs only when Analyze recording is pressed.

The file uploader accepts WAV, MP3, FLAC and OGG. The existing-recording selector
is immediately below it. Uploading a new file clears the selected example and
old output. Selecting an example recreates the uploader with a fresh widget key,
clearing any prior uploaded file. Both paths clear the old result, preventing a
new recording from appearing next to the previous recording's prediction.

The upload arrives as bytes. `run_analysis` puts those bytes in a uniquely named
temporary directory, calls the existing analysis functions, renders the figures
to PNG bytes, and deletes the temporary directory even on errors. Session state
holds the displayed result and original audio playback bytes. Uploaded audio is
never stored in the application-wide Streamlit cache.

A shared lock serializes analysis and plotting across sessions because Matplotlib
is not thread-safe and simultaneous audio decoding can multiply memory use.
Only public, static evaluation figures and the stylesheet are cached across
visitors. This does not create a remote inference API dependency.

## How the design was preserved

`.streamlit/config.toml` sets a dark theme, near-black background (#0d0d0d),
off-white text (#e8e8e3), dull teal accent (#5a8a80), sharp corners, and borders.
`assets/streamlit.css` styles the hero, upload area, tabs, tables and result cards.
The bundled IBM Plex Mono fonts are embedded in the stylesheet, so the browser
does not need to contact a font service. Matplotlib uses those same font files.

The Analyze button is solid off-white with dark text to make the next action
clear. The upload outline is teal. Results use amber (#d4a24e) for AI and teal for
human. Contributions remain thin rectangular bars. Numeric HTML table cells are
right-aligned and all dynamic cell content is HTML-escaped. Desktop columns have
a generous gap; phone layouts stack the input and result areas.

The wording was retained apart from necessary corrections: the technology credit
now says Streamlit; uploads are described as processed on the app server; and the
legend says teal/amber to match the actual bars. Streamlit supplies its own native
upload and selection control wording.

## Dependencies and hosting preparation

Streamlit replaces Gradio in pyproject.toml and uv.lock. The Gradio dependency
and its unused transitive dependencies were removed. Community Cloud understands
uv.lock directly, so no second dependency list needs to be maintained. Python
3.12 is recorded in .python-version and in the deployment instructions.

The saved model and evaluation artifacts are sufficient for the hosted app.
Training audio is not required. Local examples appear on this machine because
the files exist here; they will not appear in a clean cloud checkout unless audio
is deliberately included later. See DEPLOYMENT.md for the exact publishing steps.

## Verification

Focused Streamlit AppTest checks exercised the three tabs, empty submission,
selection and analysis of all three existing examples, and clearing prior output
when selection changes. Both an actual encoded upload and a too-short WAV were
passed through the new upload entrypoint. All example scores matched baseline.py:

- Human reference: 0.4226964944251506.
- Generated example: 0.8330023530948172.
- Known human false positive: 0.7980986156369996.

Both plots and all ten feature rows were returned for valid input. Short input
cleared playback, plots and feature values. The model's source-integrity hashes
also match the committed processing files, supporting an unchanged Linux checkout.
The local server returned a healthy response; desktop and mobile layouts were
rendered for visual review. A real browser upload also returned the expected
human score, and the evaluation tab was opened successfully. No additional
model-training or dataset test stage was
introduced.

## Limits and article framing

This is still an exploratory, direct-file detector. The AI score is not a
confidence percentage or proof of authorship. Microphone robustness, unseen
music generators and an inconclusive outcome remain outside this version.

For the article, distinguish model development from interface engineering:
we trained a small interpretable classifier, exposed its exact weighted feature
contributions, then migrated its presentation without changing its predictions.
The framework supplies the interface; the model remains our trained model.

Public deployment is not completed by these local checks. The first Linux cloud
build and hosted resource usage must still be observed when the user publishes.
