"""Synthetic signals check behavior, not AI-detection ability."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf

from audio_inspection import inspect_audio, load_audio, plot_audio, spectrogram


@pytest.mark.parametrize("channels", [1, 2])
def test_load_preserves_channels_and_metadata(tmp_path, channels):
    path = tmp_path / "tone.wav"
    samples = np.zeros((8000, channels), dtype=np.float32)
    samples[:, 0] = 0.25
    sf.write(path, samples, 8000, subtype="FLOAT")

    audio = load_audio(path)

    assert audio.samples.shape == (8000, channels)
    np.testing.assert_allclose(audio.samples, samples)
    assert audio.sample_rate == 8000
    assert audio.channels == channels
    assert audio.duration_seconds == 1.0


def test_spectrogram_finds_known_tone(tmp_path):
    rate = 8000
    time = np.arange(rate) / rate
    path = tmp_path / "tone.wav"
    sf.write(path, 0.5 * np.sin(2 * np.pi * 1000 * time), rate)

    power, frequencies, times = spectrogram(load_audio(path))

    assert power.shape == (len(frequencies), len(times))
    peak = frequencies[np.argmax(power.mean(axis=1))]
    assert abs(peak - 1000) <= rate / 1024


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_audio(tmp_path / "missing.wav")


def test_unreadable_file(tmp_path):
    path = tmp_path / "invalid.wav"
    path.write_text("This is not audio")
    with pytest.raises(ValueError, match="Could not read"):
        load_audio(path)


@pytest.mark.parametrize("samples", [np.empty((0, 1)), np.array([[np.nan]]), np.array([[np.inf]])])
def test_invalid_samples(tmp_path, samples):
    path = tmp_path / "invalid.wav"
    sf.write(path, samples, 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="empty|non-finite"):
        load_audio(path)


@pytest.mark.parametrize("length", [1, 8000])
def test_silence_and_short_audio_produce_finite_plots(tmp_path, length):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(length), 8000)
    audio = load_audio(path)
    figure = plot_audio(audio)
    try:
        assert figure.axes[0].get_xlabel() == "Time (s)"
        assert figure.axes[0].get_ylabel() == "Amplitude"
        assert figure.axes[1].get_ylabel() == "Frequency (Hz)"
        assert np.isfinite(figure.axes[1].images[0].get_array()).all()
        figure.savefig(tmp_path / "inspection.png")
        assert (tmp_path / "inspection.png").stat().st_size > 0
    finally:
        plt.close(figure)


def test_inspection_reports_metadata(tmp_path, capsys):
    path = tmp_path / "example.wav"
    sf.write(path, np.zeros((4000, 2)), 8000)
    figure = inspect_audio(path)
    try:
        output = capsys.readouterr().out
        assert "8000 Hz" in output
        assert "(4000, 2)" in output
        assert "0.500 s" in output
    finally:
        plt.close(figure)
