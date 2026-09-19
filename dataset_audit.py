"""Metadata-only inspection of Echoes and FMA; never trains or downloads audio.

Commands:
  python dataset_audit.py list echoes
  python dataset_audit.py fetch echoes path/inside/archive.csv
  python dataset_audit.py report echoes.csv tracks.csv

Source metadata and transfer receipts stay in data/source_metadata. Only exact
title/artist matches and conservative normalized matches are linked; no fuzzy
match is automatically accepted. A match is not license or provenance clearance.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen
import zipfile


ECHOES_REVISION = "14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2"
SOURCES = {
    "echoes": f"https://huggingface.co/datasets/Octavian97/Echoes/resolve/{ECHOES_REVISION}/Echoes.zip",
    "fma": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}
METADATA_DIR = Path("data/source_metadata")


class TransferBudget:
    """Persist actual response-body bytes against a cumulative 400 MiB cap.

    A server must honor byte ranges; a full-archive response is closed before
    reading its body. Network protocol overhead is not counted in this ledger.
    """

    def __init__(self, ledger: Path, limit: int = 400 * 1024**2):
        self.ledger = ledger
        self.limit = limit
        self.entries = json.loads(ledger.read_text()) if ledger.exists() else []
        self.used = sum(entry["bytes"] for entry in self.entries)

    def fetch(self, url: str, start: int, end: int) -> tuple[bytes, int]:
        length = end - start + 1
        if start < 0 or length < 1 or self.used + length > self.limit:
            raise ValueError("Invalid range or metadata transfer budget exceeded")
        request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=45) as response:
            if response.status != 206:
                raise ValueError("Server did not honor range request; full download refused")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[2])) != (start, end):
                raise ValueError("Server returned an unexpected byte range")
            chunks = []
            received = 0
            try:
                while received < length:
                    chunk = response.read(min(1024**2, length - received))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    received += len(chunk)
            finally:
                self.used += received
                self.entries.append({"url": url, "start": start, "end": end, "bytes": received})
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                self.ledger.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
            if received != length:
                raise ValueError("Truncated metadata range response")
            return b"".join(chunks), int(match[3])


class RangeReader(io.RawIOBase):
    """Make a remote ZIP seekable using only explicitly requested byte ranges."""

    def __init__(self, size: int, fetch):
        self.size, self.fetch, self.position = size, fetch, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = (0, self.position, self.size)[whence] + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        data = self.fetch(self.position, end - 1)
        self.position = end
        return data


def remote_zip(source: str, budget: TransferBudget) -> zipfile.ZipFile:
    url = SOURCES[source]
    _, size = budget.fetch(url, 0, 0)
    return zipfile.ZipFile(RangeReader(size, lambda start, end: budget.fetch(url, start, end)[0]))


def normalize(value: str) -> str:
    """Normalize Unicode, case and whitespace, preserving punctuation."""
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def read_fma(path: Path) -> list[dict[str, str]]:
    """Read FMA's two-row column header and third-row index label without pandas."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        groups, names = next(reader), next(reader)
        columns = list(zip(groups, names))
        wanted = {"title": ("track", "title"), "artist": ("artist", "name"),
                  "artist_id": ("artist", "id"), "subset": ("set", "subset"),
                  "license": ("track", "license")}
        try:
            indices = {key: columns.index(column) for key, column in wanted.items()}
        except ValueError as error:
            raise ValueError("FMA metadata is missing required columns") from error
        rows = []
        for row in reader:
            if not row or row[0] == "track_id":
                continue
            rows.append({"track_id": row[0], **{key: row[index] for key, index in indices.items()}})
        return rows


def audit(echoes: list[dict[str, str]], fma: list[dict[str, str]]) -> list[dict[str, str]]:
    """Link reference strings to full FMA metadata before checking small membership."""
    exact, normalized = defaultdict(list), defaultdict(list)
    for track in fma:
        key = f"{track['title']} - {track['artist']}"
        exact[key].append(track)
        normalized[normalize(key)].append(track)
    result = []
    required = {"original_audio", "type", "generator", "genre", "path_in_dataset"}
    for row in echoes:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Echoes missing columns: {', '.join(sorted(missing))}")
        if row["type"] not in {"TTA", "ATA"}:
            raise ValueError(f"Unknown generation type: {row['type']!r}")
        reference = row["original_audio"]
        matches = exact.get(reference, [])
        status = "exact"
        if not matches:
            matches = normalized.get(normalize(reference), [])
            status = "normalized"
        if not matches:
            status = "missing"
        elif len(matches) > 1:
            status = "ambiguous"
        unique = len(matches) == 1
        result.append({**row, "match_status": status,
                       "fma_track_ids": ";".join(track["track_id"] for track in matches),
                       "fma_small": ("yes" if matches[0]["subset"] == "small" else "no") if unique else "unknown",
                       "fma_artist_id": matches[0]["artist_id"] if unique else "",
                       "fma_license": matches[0]["license"] if unique else ""})
    return result


def summarize(rows: list[dict[str, str]]) -> dict:
    tta = [row for row in rows if row["type"] == "TTA"]
    paired = [row for row in tta if row["fma_small"] == "yes"]
    references = {row["original_audio"]: row for row in rows}
    small_tracks = {row["fma_track_ids"]: row for row in paired}
    paths = Counter(row["path_in_dataset"] for row in rows)
    no_derivatives = {key for key, row in small_tracks.items()
                      if "derivative" in row["fma_license"].casefold()}
    # This only removes two visible defects; it is NOT license clearance or
    # proof of authorship. Empty/unclear license terms still require review.
    provisional = [row for row in paired if row["fma_track_ids"] not in no_derivatives
                   and paths[row["path_in_dataset"]] == 1]
    return {
        "total_rows": len(rows), "tta_rows": len(tta),
        "unique_references": len(references),
        "unique_reference_match_status": dict(sorted(Counter(row["match_status"] for row in references.values()).items())),
        "unique_reference_small_status": dict(sorted(Counter(row["fma_small"] for row in references.values()).items())),
        "duplicate_paths": {path: count for path, count in sorted(paths.items()) if count > 1},
        "duplicate_path_extra_rows": sum(count - 1 for count in paths.values()),
        "small_unique_artists": len({row["fma_artist_id"] for row in small_tracks.values()}),
        "small_licenses_by_unique_track": dict(sorted(Counter(row["fma_license"] for row in small_tracks.values()).items())),
        "small_no_derivatives_track_ids": sorted(no_derivatives),
        "tta_small_non_nd_unique_path_rows": len(provisional),
        "tta_small_non_nd_unique_path_references": len({row["fma_track_ids"] for row in provisional}),
        "tta_small_non_nd_unique_path_by_generator": dict(sorted(Counter(row["generator"] for row in provisional).items())),
        "tta_small_non_nd_unique_path_by_genre": dict(sorted(Counter(row["genre"] for row in provisional).items())),
        "match_status_rows": dict(sorted(Counter(row["match_status"] for row in rows).items())),
        "tta_unique_small_matches": len({row["fma_track_ids"] for row in paired}),
        "tta_small_matched_rows": len(paired),
        "tta_by_generator": dict(sorted(Counter(row["generator"] for row in tta).items())),
        "tta_by_genre": dict(sorted(Counter(row["genre"] for row in tta).items())),
        "tta_small_by_generator": dict(sorted(Counter(row["generator"] for row in paired).items())),
        "tta_small_by_genre": dict(sorted(Counter(row["genre"] for row in paired).items())),
        "matched_small_licenses": dict(sorted(Counter(row["fma_license"] for row in paired).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("source", choices=SOURCES)
        if name == "fetch":
            command.add_argument("member")
    report = commands.add_parser("report")
    report.add_argument("echoes_csv", type=Path)
    report.add_argument("fma_csv", type=Path)
    args = parser.parse_args()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.command in {"list", "fetch"}:
        budget = TransferBudget(METADATA_DIR / "transfers.json")
        with remote_zip(args.source, budget) as archive:
            if args.command == "list":
                listing = [{"name": item.filename, "bytes": item.file_size,
                            "compressed_bytes": item.compress_size} for item in archive.infolist()]
                (METADATA_DIR / f"{args.source}_archive_index.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
                for item in listing:
                    if Path(item["name"]).suffix.lower() in {".csv", ".json", ".txt", ".md"}:
                        print(item)
                print(f"Archive entries: {len(listing)}")
            else:
                item = archive.getinfo(args.member)
                if Path(item.filename).suffix.lower() not in {".csv", ".json", ".txt", ".md"}:
                    raise ValueError("Only metadata members may be fetched")
                if item.file_size > 256 * 1024**2:
                    raise ValueError("Metadata member exceeds 256 MiB uncompressed limit")
                target = METADATA_DIR / f"{args.source}_{Path(item.filename).name}"
                if target.exists():
                    raise ValueError(f"Refusing to overwrite {target}")
                data = archive.read(item)
                target.write_bytes(data)
                receipt = {"source_url": SOURCES[args.source], "member": item.filename,
                           "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                target.with_suffix(target.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                print(json.dumps(receipt, indent=2))
        print(f"Cumulative metadata transfer: {budget.used} / {budget.limit} bytes")
    else:
        with args.echoes_csv.open(encoding="utf-8-sig", newline="") as handle:
            echoes = list(csv.DictReader(handle))
        rows = audit(echoes, read_fma(args.fma_csv))
        if not rows:
            raise ValueError("Echoes metadata is empty")
        with (METADATA_DIR / "reference_matches.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        output = json.dumps(summarize(rows), indent=2)
        (METADATA_DIR / "summary.json").write_text(output + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
