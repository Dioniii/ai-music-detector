"""Download and inspect only the six selected local pilot files.

Run `python pilot_audio.py download` or `python pilot_audio.py inspect`.
This is a data sanity check, not a training or detection pipeline.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_inspection import load_audio, plot_audio
from tools.dataset_audit import RangeReader, SOURCES, TransferBudget


AUDIO_DIR = Path("data/audio_pilot")
PLOTS_DIR = Path("data/pilot_plots")
MANIFEST = Path("data/preparation/audio_pilot_manifest.csv")
ARCHIVES = {"fma_small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
            "echoes": SOURCES["echoes"]}
SELECTED = {
    "human_45101": ("fma_small", "fma_small/045/045101.mp3"),
    "human_127294": ("fma_small", "fma_small/127/127294.mp3"),
    "human_112315": ("fma_small", "fma_small/112/112315.mp3"),
    "ai_45101": ("echoes", "Echoes/TTA/acestep/Straw_Fields_Rolemusic_acestep_TTA_001.mp3"),
    "ai_127294": ("echoes", "Echoes/TTA/suno/Autopsy_Oh_Yeah_the_Future_suno_TTA_001.mp3"),
    "ai_112315": ("echoes", "Echoes/TTA/udio/Digital_Lightning_Cloudkicker_udio_TTA_001.mp3"),
}


def download_member(
    archive: zipfile.ZipFile, member: str, expected_bytes: int,
    expected_compressed: int, target: Path, allowed_members: set[str],
) -> str:
    """Read a selected member, check its sizes/CRC, write exclusively, return SHA-256."""
    if member not in allowed_members:
        raise ValueError("Archive member is not selected for this pilot")
    if target.exists():
        raise FileExistsError(target)
    info = archive.getinfo(member)
    if (info.file_size != expected_bytes or info.compress_size != expected_compressed
            or info.file_size > 8 * 1024**2):
        raise ValueError("Unexpected archive member size")
    data = archive.read(info)  # zipfile checks CRC; never extract archive paths.
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def inspect_file(path: Path, expected_duration: float) -> dict:
    """Measure decoded samples; near-zero/full-scale ratios are not forensic claims."""
    try:
        audio = load_audio(path)
    except (FileNotFoundError, ValueError) as error:
        return {"status": "decode_error", "error": str(error)}
    magnitude = np.abs(audio.samples)
    return {
        "status": "ok", "sample_rate": audio.sample_rate,
        "shape": list(audio.samples.shape), "channels": audio.channels,
        "duration_seconds": audio.duration_seconds,
        "expected_duration_seconds": expected_duration,
        "duration_matches": abs(audio.duration_seconds - expected_duration) <= 0.25,
        "duration_tolerance_seconds": 0.25,
        "peak_amplitude": float(magnitude.max()),
        "rms_amplitude": float(np.sqrt(np.mean(np.square(audio.samples, dtype=np.float64)))),
        "channel_mean_amplitude": np.mean(audio.samples, axis=0, dtype=np.float64).tolist(),
        "all_zero": bool(np.all(audio.samples == 0)),
        "near_zero_fraction": float(np.mean(magnitude < 1e-4)),
        "at_or_above_full_scale_fraction": float(np.mean(magnitude >= 1.0)),
    }


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6 or {row["sample_id"] for row in rows} != set(SELECTED):
        raise ValueError("Manifest must contain exactly the six selected sample IDs")
    for row in rows:
        if (row["archive"], row["archive_member"]) != SELECTED[row["sample_id"]]:
            raise ValueError("Manifest refers to an unselected archive member")
    return rows


def download(rows: list[dict[str, str]]) -> None:
    budget = TransferBudget(Path("data/source_metadata/pilot_audio_transfers.json"),
                            limit=32 * 1024**2)
    for source, url in ARCHIVES.items():
        subset = [row for row in rows if row["archive"] == source]
        _, size = budget.fetch(url, 0, 0)
        with zipfile.ZipFile(RangeReader(size, lambda start, end: budget.fetch(url, start, end)[0])) as archive:
            allowed = {member for archive_source, member in SELECTED.values() if archive_source == source}
            for row in subset:
                target = AUDIO_DIR / f"{row['sample_id']}.mp3"
                receipt_path = target.with_suffix(".receipt.json")
                if target.exists() and receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if (receipt["source_url"] != url or receipt["archive_member"] != row["archive_member"]
                            or hashlib.sha256(target.read_bytes()).hexdigest() != receipt["sha256"]):
                        raise ValueError(f"Existing audio receipt mismatch: {target}")
                    print(f"Already downloaded and hash checked: {target}", flush=True)
                    continue
                digest = download_member(archive, row["archive_member"], int(row["file_bytes"]),
                                         int(row["compressed_bytes"]), target, allowed)
                receipt = {"sample_id": row["sample_id"], "source_url": url,
                           "archive_member": row["archive_member"], "sha256": digest,
                           "file_bytes": target.stat().st_size}
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
                print(f"Downloaded: {target}", flush=True)
    print(f"Pilot response-body transfer: {budget.used} / {budget.limit} bytes")


def inspect(rows: list[dict[str, str]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        path = AUDIO_DIR / f"{row['sample_id']}.mp3"
        result = {"sample_id": row["sample_id"], "file_path": path.as_posix(),
                  **inspect_file(path, float(row["expected_duration_seconds"]))}
        if result["status"] == "ok":
            result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            figure = plot_audio(load_audio(path))
            figure.suptitle(f"{row['sample_id']}: {row['reference']}")
            plot_path = PLOTS_DIR / f"{row['sample_id']}.png"
            figure.savefig(plot_path, dpi=100)
            plt.close(figure)
            result["plot_path"] = plot_path.as_posix()
        results.append(result)
    Path("data/preparation/audio_pilot_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "inspect"])
    args = parser.parse_args()
    rows = read_manifest()
    if args.command == "download":
        download(rows)
    else:
        inspect(rows)


if __name__ == "__main__":
    main()
