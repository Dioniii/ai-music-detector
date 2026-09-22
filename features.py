"""Ten descriptive measurements from a preprocessed section of at least 10 seconds, at 24 kHz.

Features describe audio properties. They do not establish AI generation.
"""

import librosa
import numpy as np
from numpy.typing import NDArray

from preprocessing import CLIP_SAMPLES, TARGET_SAMPLE_RATE, prepare_recording


FRAME_LENGTH = 1024
HOP_LENGTH = 512
FLATNESS_POWER_FLOOR = 1e-10
FEATURE_NAMES = tuple(
    f"{feature}_{statistic}"
    for feature in (
        "rms", "zero_crossing_rate", "spectral_centroid_hz",
        "spectral_bandwidth_hz", "spectral_flatness",
    )
    for statistic in ("mean", "std")
)


def extract_features(clip: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return ten finite values in FEATURE_NAMES order, leaving input unchanged.

    Caller must supply at least 240000 floating samples at 24000 Hz. Arrays cannot carry
    their own sample rate; use preprocess_audio to satisfy this contract.

    Frames are left-aligned with no edge padding, each 1024 samples, stepping
    by 512. Any incomplete final frame is omitted.
    RMS/ZCR use unwindowed samples; spectral features use Hann-windowed magnitude
    spectra. Centroid and bandwidth are magnitude-weighted, in Hz; bandwidth
    uses p=2. Flatness uses power, with a fixed floor to keep logs finite.

    Silence has zero RMS/ZCR/centroid/bandwidth but flatness 1 because every power
    bin is floored equally. This is a numerical convention, not evidence of noise.
    Standard deviations are population values (ddof=0). No scaling is learned.
    """
    if (not isinstance(clip, np.ndarray) or clip.ndim != 1 or len(clip) < CLIP_SAMPLES
            or not np.issubdtype(clip.dtype, np.floating)
            or not np.isfinite(clip).all()):
        raise ValueError("Expected at least 240000 finite floating-point mono samples at 24000 Hz")
    signal = clip.astype(np.float64, copy=False)
    magnitude = np.abs(librosa.stft(
        signal, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
        win_length=FRAME_LENGTH, window="hann", center=False,
    ))
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=TARGET_SAMPLE_RATE)
    frame_features = np.vstack([
        librosa.feature.rms(y=signal, frame_length=FRAME_LENGTH,
                            hop_length=HOP_LENGTH, center=False, dtype=np.float64),
        librosa.feature.zero_crossing_rate(
            signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
            center=False, threshold=0.0, zero_pos=True,
        ),
        centroid,
        librosa.feature.spectral_bandwidth(
            S=magnitude, sr=TARGET_SAMPLE_RATE, centroid=centroid, norm=True, p=2,
        ),
        librosa.feature.spectral_flatness(S=magnitude, amin=FLATNESS_POWER_FLOOR, power=2.0),
    ])
    # Two summaries per feature -> (5, 2) -> ordered vector (10,).
    means = frame_features.mean(axis=1)
    deviations = frame_features.std(axis=1, ddof=0)
    result = np.column_stack((means, deviations)).reshape(-1)
    if result.shape != (10,) or not np.isfinite(result).all():
        raise ValueError("Feature extraction did not produce ten finite measurements")
    return result


def recording_features(audio):
    """Measure every selected section, then average its ten features."""
    mono, sections = prepare_recording(audio)
    vectors = []
    for section in sections:
        start = round(section['start_seconds'] * TARGET_SAMPLE_RATE)
        end = round(section['end_seconds'] * TARGET_SAMPLE_RATE)
        vectors.append(extract_features(mono[start:end].astype(np.float32)))
    return np.mean(vectors, axis=0), sections
