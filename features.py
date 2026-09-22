"""Ten descriptive measurements from a preprocessed section of at least 10 seconds, at 24 kHz.

Features describe audio properties. They do not establish AI generation.
"""

from pathlib import Path
from functools import lru_cache

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


# EfficientAT is frozen: only the small classifier in baseline.py is trained.
ENCODER_DIR = Path(__file__).resolve().parent / 'data/encoder'
ENCODER_PATH = ENCODER_DIR / 'efficientat_mn10.pt'
ENCODER_NAMES = tuple(f'embedding_{i:03d}' for i in range(960))
ENCODER_REVISION = 'a425fdce92572e602a1d5634799bd9f1f2efa806'


def export_encoder():
    """Build a portable encoder from pinned official source; run only during setup."""
    import hashlib
    import io
    import json
    import sys
    import urllib.request
    import zipfile
    from pathlib import Path
    import torch

    if ENCODER_PATH.exists():
        raise FileExistsError('Encoder already exported; refusing to overwrite it')
    source_dir = ENCODER_DIR.parent / 'encoder_source'
    root = source_dir / f'EfficientAT-{ENCODER_REVISION}'
    if not root.exists():
        url = f'https://codeload.github.com/fschmid56/EfficientAT/zip/{ENCODER_REVISION}'
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.infolist():
            if not (source_dir / member.filename).resolve().is_relative_to(source_dir.resolve()):
                raise ValueError('Unexpected source archive path')
        archive.extractall(source_dir)
    sys.path.insert(0, str(root))
    import contextlib
    with contextlib.chdir(root):
        from models.mn.model import get_model
        from models.preprocess import AugmentMelSTFT
    checkpoint = source_dir / 'mn10_as.pt'
    url = 'https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as_mAP_471.pt'
    if not checkpoint.exists():
        torch.hub.download_url_to_file(url, str(checkpoint))
    torch.set_num_threads(2)
    # Suppress the upstream constructor's lengthy architecture printout.
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        model = get_model(pretrained_name=None).eval()
        mel = AugmentMelSTFT().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = mel
            self.features = model.features

        def forward(self, waveform):
            x = self.features(self.mel(waveform).unsqueeze(1))
            return x.mean(dim=(2, 3))

    encoder = Embedding().eval()
    with torch.inference_mode():
        example = torch.zeros(1, 320000)
        traced = torch.jit.trace(encoder, example, check_trace=False)
        traced = torch.jit.freeze(traced)
        # Different input lengths must retain the official encoder's behavior.
        torch.manual_seed(42)
        for seconds in (10, 20):
            waveform = torch.randn(1, seconds * 32000) * .05
            expected = model(mel(waveform).unsqueeze(1))[1]
            torch.testing.assert_close(traced(waveform), expected, rtol=1e-4, atol=1e-5)
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    traced.save(str(ENCODER_PATH))
    (ENCODER_DIR / 'LICENSE.txt').write_text((root / 'LICENSE').read_text(), encoding='utf-8')
    provenance = {
        'model': 'EfficientAT mn10_as', 'source_revision': ENCODER_REVISION,
        'checkpoint_url': url, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'export_sha256': hashlib.sha256(ENCODER_PATH.read_bytes()).hexdigest(),
        'sample_rate': 32000, 'embedding_dimensions': 960,
        'preprocessing': 'Official evaluation mel frontend; no augmentation',
        'pooling': 'Mean over final convolution frequency/time dimensions, then mean over sections',
        'torch_version': torch.__version__, 'export_parity': 'Passed at 10 and 20 seconds',
    }
    (ENCODER_DIR / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print(f'Exported {ENCODER_PATH} ({ENCODER_PATH.stat().st_size/1024**2:.1f} MiB)')


@lru_cache(maxsize=1)
def load_encoder():
    """Share one frozen CPU encoder across predictions; no network requests."""
    import hashlib
    import json
    import torch
    provenance = json.loads((ENCODER_DIR / 'provenance.json').read_text())
    if hashlib.sha256(ENCODER_PATH.read_bytes()).hexdigest() != provenance['export_sha256']:
        raise ValueError('Encoder artifact does not match its recorded checksum')
    torch.set_num_threads(2)
    return torch.jit.load(str(ENCODER_PATH), map_location='cpu').eval()


def encoder_features(audio):
    """Average 960 learned features from the same random-section policy at 32 kHz."""
    import torch
    encoder = load_encoder()
    mono, sections = prepare_recording(audio, sample_rate=32000)
    vectors = []
    with torch.inference_mode():
        for section in sections:
            start = round(section['start_seconds'] * 32000)
            end = round(section['end_seconds'] * 32000)
            waveform = torch.from_numpy(mono[start:end].astype(np.float32)).unsqueeze(0)
            vectors.append(encoder(waveform).squeeze(0).numpy().copy())
    vector = np.mean(vectors, axis=0, dtype=np.float64)
    if vector.shape != (960,) or not np.isfinite(vector).all():
        raise ValueError('Encoder did not return 960 finite features')
    return vector, sections


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export the frozen EfficientAT encoder once.')
    parser.add_argument('command', choices=['export-encoder'])
    parser.parse_args()
    export_encoder()
