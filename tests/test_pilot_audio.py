"""Behavior tests for the six-file audio check; no network required."""

import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from pilot_audio import download_member, inspect_file


def archive_with(name, content):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    buffer.seek(0)
    return zipfile.ZipFile(buffer)


def test_only_selected_archive_members_can_be_downloaded(tmp_path):
    with archive_with("other.mp3", b"audio") as archive:
        with pytest.raises(ValueError, match="selected"):
            download_member(archive, "other.mp3", 5, 5, tmp_path / "audio.mp3", {"selected.mp3"})
    assert not list(tmp_path.iterdir())


def test_unexpected_archive_size_is_rejected_before_reading(tmp_path):
    with archive_with("selected.mp3", b"audio") as archive:
        with pytest.raises(ValueError, match="size"):
            download_member(archive, "selected.mp3", 100, 100, tmp_path / "audio.mp3", {"selected.mp3"})
    assert not list(tmp_path.iterdir())


def test_download_writes_only_named_file_and_never_overwrites(tmp_path):
    target = tmp_path / "audio.mp3"
    with archive_with("selected.mp3", b"audio") as archive:
        download_member(archive, "selected.mp3", 5, 5, target, {"selected.mp3"})
        assert target.read_bytes() == b"audio"
        with pytest.raises(FileExistsError):
            download_member(archive, "selected.mp3", 5, 5, target, {"selected.mp3"})


def test_inspection_reports_shape_duration_and_signal_properties(tmp_path):
    path = tmp_path / "tone.wav"
    time = np.arange(8000) / 8000
    signal = 0.5 * np.sin(2 * np.pi * 440 * time)
    sf.write(path, np.column_stack([signal, signal]), 8000, subtype="FLOAT")
    result = inspect_file(path, expected_duration=1.0)
    assert result["status"] == "ok"
    assert result["shape"] == [8000, 2]
    assert result["duration_seconds"] == 1.0
    assert result["peak_amplitude"] == pytest.approx(0.5)
    assert result["rms_amplitude"] == pytest.approx(0.5 / np.sqrt(2))
    assert result["duration_matches"] is True
    assert result["all_zero"] is False


def test_silence_and_duration_mismatch_are_visible(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(8000), 8000)
    result = inspect_file(path, expected_duration=30.0)
    assert result["status"] == "ok"
    assert result["all_zero"] is True
    assert result["duration_matches"] is False


def test_bad_audio_is_reported_instead_of_disappearing(tmp_path):
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"not audio")
    result = inspect_file(path, expected_duration=30.0)
    assert result["status"] == "decode_error"
    assert result["error"]


def test_inspection_measures_channel_offsets_without_changing_audio(tmp_path):
    path = tmp_path / "offset.wav"
    sf.write(path, np.tile([0.25, -0.5], (8000, 1)), 8000, subtype="FLOAT")
    original = path.read_bytes()
    result = inspect_file(path, expected_duration=1.0)
    assert result["channel_mean_amplitude"] == pytest.approx([0.25, -0.5])
    assert path.read_bytes() == original
