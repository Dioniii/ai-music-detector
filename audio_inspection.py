"""Inspect local audio; these measurements do not detect AI generation.

Run: python audio_inspection.py path/to/audio.wav
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import soundfile as sf


@dataclass(frozen=True)
class Audio:
    """Samples always have shape (frames, channels), including mono audio."""

    samples: NDArray[np.float32]
    sample_rate: int

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def load_audio(path: str | Path) -> Audio:
    """Read samples without resampling, mixing channels, or normalizing volume.

    Loads the whole file into memory; use short WAV files for this lesson.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as error:
        raise ValueError(f"Could not read audio file: {path}") from error
    if samples.shape[0] == 0:
        raise ValueError("Audio is empty.")
    if not np.isfinite(samples).all():
        raise ValueError("Audio contains non-finite samples.")
    return Audio(samples=samples, sample_rate=rate)


def recording_quality_issue(audio: Audio) -> str | None:
    """Conservative UI guard, not a noise estimator or authorship feature.

    Require at least five seconds in one-second blocks at >=-50 dBFS AC RMS.
    Reject if any channel has >=10% of samples at >=0.999 full scale. These are
    initial heuristic limits, not thresholds validated on phone recordings.
    Scan in blocks to avoid copying a whole decoded recording. Do not modify it.
    """
    audible_seconds = 0.0
    clipped = np.zeros(audio.channels, dtype=np.int64)
    for start in range(0, len(audio.samples), audio.sample_rate):
        block = audio.samples[start:start + audio.sample_rate]
        clipped += np.count_nonzero(np.abs(block) >= 0.999, axis=0)
        mono = block.mean(axis=1, dtype=np.float64)
        # A constant electrical offset is not audible music.
        mono -= mono.mean()
        rms = float(np.sqrt(np.mean(mono * mono)))
        if rms >= 10 ** (-50 / 20):
            audible_seconds += len(block) / audio.sample_rate
    if np.any(clipped / len(audio.samples) >= 0.10):
        return ('The recording may be too loud or distorted. Move the microphone '
                'farther from the speaker or turn the music down, then record again. '
                'For an uploaded file, try a different recording.')
    if audible_seconds < 5.0:
        return ('The recording is too quiet or has too little clear sound to analyze reliably. Move closer to the '
                'music and record again. Check that the microphone is not covered. '
                'For an uploaded file, try a clearer recording.')
    return None


def spectrogram(
    audio: Audio,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return first-channel power density, frequencies (Hz), and times (s).

    Use 1024-sample Hann windows with 50% overlap. Shorter recordings
    are zero-padded for display; padding does not add frequency information.
    """
    channel = audio.samples[:, 0]
    if len(channel) < 1024:
        channel = np.pad(channel, (0, 1024 - len(channel)))
    return mlab.specgram(
        channel,
        NFFT=1024,
        Fs=audio.sample_rate,
        noverlap=512,
        window=mlab.window_hanning,
        mode="psd",
        scale_by_freq=True,
    )


def plot_audio(audio: Audio) -> Figure:
    """Create first-channel plots; the caller owns displaying/closing them."""
    time = np.arange(audio.samples.shape[0]) / audio.sample_rate
    power, frequencies, times = spectrogram(audio)
    # A fixed floor avoids log10(0) for silence. These are not acoustic dB SPL.
    decibels = 10 * np.log10(np.maximum(power, 1e-12))
    figure, (waveform_axis, spectrum_axis) = plt.subplots(
        2, 1, figsize=(10, 6), layout="constrained"
    )
    waveform_axis.plot(time, audio.samples[:, 0], linewidth=0.7)
    waveform_axis.set(
        title="Waveform — channel 1", xlabel="Time (s)", ylabel="Amplitude"
    )
    half_hop_seconds = 256 / audio.sample_rate
    heatmap = spectrum_axis.imshow(
        decibels,
        origin="lower",
        aspect="auto",
        extent=(
            times[0] - half_hop_seconds,
            times[-1] + half_hop_seconds,
            frequencies[0],
            frequencies[-1],
        ),
        cmap="magma",
        interpolation="nearest",
    )
    spectrum_axis.set(
        title="Spectrogram — channel 1 (1024-sample window)",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )
    figure.colorbar(heatmap, ax=spectrum_axis, label="Power density (dB re 1 amplitude²/Hz)")
    return figure


def inspect_audio(path: str | Path) -> Figure:
    """Print metadata and return the plots for one local audio file."""
    audio = load_audio(path)
    print(f"Sample rate: {audio.sample_rate} Hz")
    print(f"Channels: {audio.channels}")
    print(f"Shape (frames, channels): {audio.samples.shape}")
    print(f"Duration: {audio.duration_seconds:.3f} s")
    return plot_audio(audio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a short local WAV file")
    args = parser.parse_args()
    try:
        inspect_audio(args.path)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    plt.show()


if __name__ == "__main__":
    main()
