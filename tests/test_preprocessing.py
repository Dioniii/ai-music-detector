"""Signal tests check the preprocessing contract, not detector accuracy."""

import numpy as np
import pytest

from audio_inspection import Audio
from preprocessing import preprocess_audio


def tone(rate, frequency, seconds=11):
    time = np.arange(rate * seconds) / rate
    return (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


@pytest.mark.parametrize("rate", [16000, 24000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_output_contract_and_frequency(rate, channels):
    signal = tone(rate, 1000)
    samples = np.repeat(signal[:, None], channels, axis=1)
    original = samples.copy()
    clip = preprocess_audio(Audio(samples, rate), start_seconds=0.5)
    assert clip.shape == (240000,)
    assert clip.dtype == np.float32
    assert np.isfinite(clip).all()
    peak = np.fft.rfftfreq(len(clip), 1 / 24000)[np.argmax(abs(np.fft.rfft(clip)))]
    assert peak == pytest.approx(1000, abs=0.1)
    assert np.sqrt(np.mean(clip**2)) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    np.testing.assert_array_equal(samples, original)


def test_stereo_mean_and_exact_start_at_target_rate():
    left = np.arange(264000, dtype=np.float32) / 264000
    right = np.zeros_like(left)
    clip = preprocess_audio(Audio(np.column_stack([left, right]), 24000), start_seconds=1)
    np.testing.assert_array_equal(clip, left[24000:264000] / 2)


def test_start_is_rounded_down_to_target_sample_grid():
    samples = np.arange(264000, dtype=np.float32)[:, None] / 264000
    clip = preprocess_audio(Audio(samples, 24000), start_seconds=0.12345)
    start = 2962  # floor(0.12345 * 24000)
    np.testing.assert_array_equal(clip, samples[start:start + 240000, 0])


def test_resampling_filters_high_frequency_alias():
    # 18 kHz would alias to 6 kHz if 48 kHz samples were simply decimated.
    signal = tone(48000, 18000)
    clip = preprocess_audio(Audio(signal[:, None], 48000), start_seconds=0.5)
    assert np.sqrt(np.mean(clip**2)) < 0.005


def test_opposite_channels_cancel_and_silence_stays_valid():
    signal = tone(24000, 440, seconds=10)
    clip = preprocess_audio(Audio(np.column_stack([signal, -signal]), 24000), start_seconds=0)
    np.testing.assert_array_equal(clip, np.zeros(240000, dtype=np.float32))


def test_offset_and_amplitudes_above_one_are_not_normalized():
    samples = np.full((240000, 1), 1.2, dtype=np.float32)
    clip = preprocess_audio(Audio(samples, 24000), start_seconds=0)
    np.testing.assert_array_equal(clip, samples[:, 0])
    assert not np.shares_memory(clip, samples)


@pytest.mark.parametrize("start", [-1, float("nan"), float("inf"), 2])
def test_invalid_start(start):
    with pytest.raises(ValueError, match="start|short"):
        preprocess_audio(Audio(np.zeros((264000, 1), np.float32), 24000), start_seconds=start)


@pytest.mark.parametrize("frames", [1, 239999])
def test_short_recording_is_not_padded(frames):
    with pytest.raises(ValueError, match="short"):
        preprocess_audio(Audio(np.zeros((frames, 1), np.float32), 24000), start_seconds=0)


def test_resampler_rounding_cannot_make_a_short_recording_eligible():
    with pytest.raises(ValueError, match="short"):
        preprocess_audio(Audio(np.zeros((441000 - 1, 1), np.float32), 44100), start_seconds=0)


@pytest.mark.parametrize("samples,rate", [
    (np.zeros(240000, np.float32), 24000),
    (np.zeros((0, 1), np.float32), 24000),
    (np.zeros((240000, 3), np.float32), 24000),
    (np.zeros((240000, 1), np.int16), 24000),
    (np.full((240000, 1), np.nan, np.float32), 24000),
    (np.full((240000, 1), np.inf, np.float32), 24000),
    (np.zeros((240000, 1), np.float32), 0),
    (np.zeros((240000, 1), np.float32), 24000.5),
])
def test_invalid_audio_contract(samples, rate):
    with pytest.raises(ValueError):
        preprocess_audio(Audio(samples, rate), start_seconds=0)
