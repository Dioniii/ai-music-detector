"""A fixed, explicit clip contract for the first baseline experiment.

Input: decoded floating-point samples shaped (frames, 1 or 2), plus sample rate.
Output: a new float32 array shaped (240000,), mono at 24000 Hz for 10 seconds.
This module neither chooses recordings nor assigns dataset splits.
"""

from math import floor, gcd, isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from audio_inspection import Audio


TARGET_SAMPLE_RATE = 24000
CLIP_SECONDS = 10
CLIP_SAMPLES = TARGET_SAMPLE_RATE * CLIP_SECONDS


def preprocess_audio(audio: Audio, *, start_seconds: float) -> NDArray[np.float32]:
    """Average channels, resample the whole recording, then take the requested clip.

    start_seconds is required, finite and nonnegative. Round it DOWN to the
    target-rate sample grid. Reject recordings too short for the requested
    start plus ten seconds, even if resampling would round their length upward.

    Resampling uses a Kaiser-window low-pass FIR filter via resample_poly.
    Its zero-extension boundary assumption can create small edge transients.
    It does not pad short recordings into eligible clips. Averaging stereo
    channels can cancel opposing signals; silence is valid and is not skipped.

    No loudness normalization, DC-offset removal, peak clipping, or file writes.
    Input arrays stay unchanged. This simple implementation holds the whole
    recording in memory; streaming long recordings is outside this increment.
    """
    samples = audio.samples
    rate = audio.sample_rate
    if (not isinstance(rate, Integral) or isinstance(rate, bool) or rate <= 0):
        raise ValueError("sample_rate must be a positive integer")
    if (not isinstance(samples, np.ndarray) or samples.ndim != 2
            or samples.shape[0] == 0 or samples.shape[1] not in (1, 2)):
        raise ValueError("samples must have shape (nonempty frames, 1 or 2 channels)")
    if not np.issubdtype(samples.dtype, np.floating) or not np.isfinite(samples).all():
        raise ValueError("samples must contain finite floating-point amplitudes")
    if (not isinstance(start_seconds, Real) or isinstance(start_seconds, bool)
            or not isfinite(start_seconds) or start_seconds < 0):
        raise ValueError("start_seconds must be a finite nonnegative number")
    if start_seconds + CLIP_SECONDS > samples.shape[0] / rate:
        raise ValueError("Audio is too short for the requested start plus 10 seconds")

    # Axis 1 contains channels, so averaging it leaves a one-dimensional signal.
    mono = samples.mean(axis=1, dtype=np.float64)
    if rate != TARGET_SAMPLE_RATE:
        divisor = gcd(int(rate), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono, up=TARGET_SAMPLE_RATE // divisor, down=int(rate) // divisor,
            window=("kaiser", 5.0), padtype="constant", cval=0.0,
        )

    start = floor(start_seconds * TARGET_SAMPLE_RATE)
    clip = np.array(mono[start:start + CLIP_SAMPLES], dtype=np.float32, copy=True)
    if clip.shape != (CLIP_SAMPLES,) or not np.isfinite(clip).all():
        raise ValueError("Preprocessing did not produce a finite 240000-sample clip")
    return clip
