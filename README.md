# AI Music Detector

Record nearby music or upload an audio file to explore whether it resembles
human-made or AI-generated music. The app returns a prediction, shows the parts
of the recording it checked, and lets you explore the model's results.

Built with **Python, EfficientAT, scikit-learn and Streamlit**.

> This is an experimental detector. Its predictions can be wrong, and its score
> is not proof of who made a song. Performance on real phone recordings has not
> been established.

## What you can do

- **Record music directly** through your browser's microphone, or upload a file.
- **Analyze the audio** and see a human/AI prediction with a score.
- **Explore the recording** through a waveform, selected sections and a sound map.
- **Delete and record again** if you want to try another clip.
- **Review the model's performance**, including its mistakes.

The interface supports desktop and mobile screens. A basic quality check asks
for another recording when the audio is very weak or heavily clipped. It cannot
reliably distinguish quiet music from background noise.

## Try it locally

You need **Python 3.12**, **uv** and **Git** installed.

```bash
git clone https://github.com/Dioniii/ai-music-detector.git
cd ai-music-detector
uv sync --locked
uv run --locked streamlit run app.py
```

Open **https://ai-musicdetector.streamlit.app/** in your browser. The trained model is included;
there is no need to download the training dataset or obtain an API key.

Choose **Use microphone** or **Upload a file**, then press **Analyze recording**.
For microphone input, record around 20-30 seconds of music and stop recording
before analyzing. Files must be **10 seconds to 5 minutes**, up to **50 MB**.
Supported upload formats are WAV, MP3, FLAC and OGG.

Microphone access requires browser permission. Localhost works on the computer
running the app; recording from a phone requires an HTTPS deployment. The
example-recording selector appears only when local example audio is available.

## How it works

```text
Recording -> selected audio sections -> sound description -> trained classifier -> result
```

The app checks up to five 20-second sections from across a recording, so it does
not rely only on the intro. Repeated analyses use the same selected sections.

**EfficientAT**, a pretrained audio model, turns each section into a numerical
description of its sound. We combine those descriptions and pass them to a
**logistic-regression classifier trained for this project**. EfficientAT stays
unchanged; our classifier learns the human/AI distinction from labeled examples.

The project started with simple measurements such as signal level and spectral
brightness. Pretrained audio descriptions performed better. The final experiment
added noise, echo and volume changes during training to better represent music
recorded through speakers and a room.

Inference runs on the app server using CPU PyTorch, without a paid inference API.
Uploads do not update the model. Temporary audio files are removed after analysis;
playback and results remain in the current session until cleared or replaced.

## Results

The dataset contains **1,000 recordings**: 500 labeled human from FMA Small and
500 labeled AI from Echoes TTA, covering 12 generators. The split is **700 for
training, 150 for validation and 150 for evaluation**, with equal class counts.
Related artists and reference recordings stay within the same split.

The final classifier trained on the 700 originals plus one altered version of
each: **1,400 examples from 700 distinct recordings**.

| Evaluation condition | Original encoder classifier | Final augmented classifier |
|---|---:|---:|
| Original audio files | 86.7% accuracy | **86.7% accuracy** |
| Simulated room noise and echo | 74.0% accuracy | **78.7% accuracy** |

Each row compares both models on the same 150 recordings. For the final model:

- On original files, **63 of 75 AI recordings were detected**, and **8 of 75 human recordings were mistaken for AI**.
- With simulated room effects, **53 of 75 AI recordings were detected**, and **10 of 75 human recordings were mistaken for AI**.

The improvement comes with tradeoffs. Clean validation accuracy fell from 88.7%
to 86.7%. On clean evaluation, the final model made fewer false accusations but
missed three more AI recordings than the original encoder classifier.

These recordings were reused across experiments, so this is a comparison
benchmark, not a fresh blind test. Simulated noise results do not establish
accuracy on real phones. See the [full experiment report](data/baseline_room/REPORT.md)
for the method, selection rule and detailed comparisons.

## What the result does not tell you

A score of **0.9 does not mean a verified 90% chance of AI authorship**. The model
has not been calibrated to make that claim. It also cannot point to a voice,
instrument or moment as proof that a song was generated.

Evaluation covers Rock and Electronic music, not every genre. Background chatter,
phone processing, unfamiliar generators and differences between dataset sources
may affect predictions. Dataset labels are not independently verified authorship,
and music combining human and AI work is outside the simple two-label setup.

## For developers

The project keeps its main workflow in six Python files:

| Files | Responsibility |
|---|---|
| `audio_inspection.py`, `preprocessing.py` | Load audio, check recording quality and select sections. |
| `features.py`, `baseline.py` | Extract audio descriptions, train the classifier and predict. |
| `showcase.py`, `app.py` | Build the charts, results and Streamlit interface. |

The current model and results are in `data/baseline_room/`; the pretrained encoder
is in `data/encoder/`. Downloaded audio, embedding caches and detailed training
arrays stay local. Dataset and noise-source metadata remain available for
reproducing the experiment.

<details>
<summary>CLI predictions, tests and training</summary>

Predict using the current model:

```bash
uv run --locked baseline.py predict "path/to/recording.mp3"
```

The CLI returns detailed JSON. The app's recording-quality gate and upload limits
are applied by the UI analysis path, not by this CLI command.

Run the tests:

```bash
uv run --locked pytest -q
```

To reproduce the final training experiment, download the fixed dataset and the
small noise collection, then choose a new output directory:

```bash
uv run --locked python -m tools.batch_audio download-expanded
uv run --locked python -m tools.batch_audio download-room-noise
uv run --locked baseline.py train --encoder --augment --output data/room_rerun
```

Training audio requires approximately 1.6 GiB, plus about 3.8 MiB for noise clips.
Downloads resume completed files. Altered audio is generated temporarily; clean
and augmented embeddings are cached locally. Only training examples fit the
scaler and classifier. Training a new model does not automatically replace the
app's default.

The original encoder classifier is preserved at
`data/baseline_encoder/model.json` for comparison. Use `--model` with the predict
command to select a different saved classifier.

</details>

## Sources and attribution

- [EfficientAT encoder](data/encoder/README.md): upstream model, license and export details.
- [Background audio](data/room_noise/README.md): ESC-50 sources, attribution and licensing.
- [Dataset manifest](data/dataset_1000.csv): music sources, labels and split assignments.
- [Final experiment](data/baseline_room/REPORT.md): results, limitations and reproduction details.

Audio sources carry their own licenses. Downloaded music and noise recordings
are not redistributed in this repository.
