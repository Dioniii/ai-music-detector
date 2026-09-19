"""Known signals check feature behavior; these are not AI-detection tests."""

import numpy as np
import pytest

from features import FEATURE_NAMES, extract_features


def tone(frequency=750, amplitude=0.5):
    time = np.arange(240000) / 24000
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def named(clip):
    return dict(zip(FEATURE_NAMES, extract_features(clip), strict=True))


def test_feature_order_shape_and_input_preservation():
    expected = (
        "rms_mean", "rms_std", "zero_crossing_rate_mean", "zero_crossing_rate_std",
        "spectral_centroid_hz_mean", "spectral_centroid_hz_std",
        "spectral_bandwidth_hz_mean", "spectral_bandwidth_hz_std",
        "spectral_flatness_mean", "spectral_flatness_std",
    )
    signal = tone()
    original = signal.copy()
    result = extract_features(signal)
    assert FEATURE_NAMES == expected
    assert result.shape == (10,)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(signal, original)


def test_silence_is_finite_and_flatness_floor_is_explicit():
    values = named(np.zeros(240000, np.float32))
    assert all(np.isfinite(value) for value in values.values())
    assert values["rms_mean"] == 0
    assert values["zero_crossing_rate_mean"] == 0
    assert values["spectral_centroid_hz_mean"] == 0
    assert values["spectral_bandwidth_hz_mean"] == 0
    # Equal floored power bins yield 1; this is NOT evidence that silence is noise.
    assert values["spectral_flatness_mean"] == pytest.approx(1)


def test_louder_tone_has_expected_rms_and_same_spectral_center():
    quiet, loud = named(tone(amplitude=0.25)), named(tone(amplitude=0.5))
    assert loud["rms_mean"] == pytest.approx(0.5 / np.sqrt(2), rel=0.002)
    assert loud["rms_mean"] == pytest.approx(2 * quiet["rms_mean"], rel=0.002)
    assert loud["spectral_centroid_hz_mean"] == pytest.approx(quiet["spectral_centroid_hz_mean"], abs=1)


def test_higher_tone_increases_centroid_and_crossing_rate():
    low, high = named(tone(750)), named(tone(3000))
    assert low["spectral_centroid_hz_mean"] == pytest.approx(750, abs=2)
    assert high["spectral_centroid_hz_mean"] == pytest.approx(3000, abs=2)
    assert high["zero_crossing_rate_mean"] > 3 * low["zero_crossing_rate_mean"]


def test_noise_is_flatter_and_broader_than_a_tone():
    noise = np.random.default_rng(42).normal(0, 0.2, 240000).astype(np.float32)
    noise_values, tone_values = named(noise), named(tone())
    assert noise_values["spectral_flatness_mean"] > tone_values["spectral_flatness_mean"] + 0.1
    assert noise_values["spectral_bandwidth_hz_mean"] > tone_values["spectral_bandwidth_hz_mean"] + 1000


def test_energy_variation_is_reflected_in_standard_deviation():
    steady = tone()
    varying = steady.copy()
    varying[120000:] *= 0.1
    assert named(varying)["rms_std"] > named(steady)["rms_std"] + 0.1


def test_constant_offset_contributes_energy_without_zero_crossings():
    values = named(np.full(240000, -0.5, np.float32))
    assert values["rms_mean"] == pytest.approx(0.5)
    assert values["rms_std"] == pytest.approx(0)
    assert values["zero_crossing_rate_mean"] == 0


@pytest.mark.parametrize("bad", [
    np.zeros((240000, 1), np.float32), np.zeros(100, np.float32),
    np.zeros(240000, np.int16), np.full(240000, np.nan, np.float32),
    np.full(240000, np.inf, np.float32),
])
def test_invalid_clip_is_rejected(bad):
    with pytest.raises(ValueError):
        extract_features(bad)
