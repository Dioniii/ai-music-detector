"""Audit tests use invented metadata only; fixtures are not training data."""

import csv
import io
import zipfile

import pytest

from dataset_audit import RangeReader, TransferBudget, audit, read_fma, summarize


def track(track_id="1", title="Song", artist="Artist", subset="small"):
    return {"track_id": track_id, "title": title, "artist": artist,
            "artist_id": "7", "subset": subset, "license": "Attribution 4.0 International"}


def generated(reference="Song - Artist", kind="TTA", generator="example"):
    return {"original_audio": reference, "type": kind, "generator": generator,
            "genre": "Rock", "path_in_dataset": f"{kind}/{generator}/song.mp3"}


def test_match_requires_both_title_and_artist():
    rows = audit([generated()], [track(), track("2", artist="Someone else")])
    assert rows[0]["match_status"] == "exact"
    assert rows[0]["fma_track_ids"] == "1"
    assert rows[0]["fma_small"] == "yes"


def test_matches_outside_small_are_reported_not_discarded():
    row = audit([generated()], [track(subset="large")])[0]
    assert row["match_status"] == "exact"
    assert row["fma_small"] == "no"


def test_duplicate_names_stay_ambiguous_even_if_only_one_is_small():
    row = audit([generated()], [track(), track("2", subset="large")])[0]
    assert row["match_status"] == "ambiguous"
    assert row["fma_track_ids"] == "1;2"
    assert row["fma_small"] == "unknown"


def test_missing_reference_is_not_fuzzy_matched():
    row = audit([generated("Different Song - Artist")], [track()])[0]
    assert row["match_status"] == "missing"
    assert row["fma_track_ids"] == ""


def test_normalized_match_is_distinguished_from_exact():
    row = audit([generated(" song - ARTIST ")], [track()])[0]
    assert row["match_status"] == "normalized"


def test_summary_excludes_ata_and_does_not_treat_matches_as_license_clearance():
    rows = audit([generated(), generated(kind="ATA"), generated("Unknown - Person")], [track()])
    result = summarize(rows)
    assert result["total_rows"] == 3
    assert result["tta_rows"] == 2
    assert result["tta_unique_small_matches"] == 1
    assert result["tta_by_generator"] == {"example": 2}
    assert result["tta_by_genre"] == {"Rock": 2}


def test_summary_flags_duplicate_paths_and_counts_unique_human_tracks():
    rows = audit([generated(), generated()], [track()])
    result = summarize(rows)
    assert result["duplicate_path_extra_rows"] == 1
    assert result["duplicate_paths"] == {"TTA/example/song.mp3": 2}
    assert result["small_licenses_by_unique_track"] == {"Attribution 4.0 International": 1}
    assert result["tta_small_non_nd_unique_path_rows"] == 0


def test_summary_separates_no_derivatives_tracks():
    item = track()
    item["license"] = "Attribution-NoDerivatives 4.0 International"
    result = summarize(audit([generated()], [item]))
    assert result["small_no_derivatives_track_ids"] == ["1"]
    assert result["tta_small_non_nd_unique_path_rows"] == 0


def test_missing_echoes_columns_fail_clearly():
    with pytest.raises(ValueError, match="original_audio"):
        audit([{"generator": "example"}], [track()])


def test_unknown_generation_type_fails():
    with pytest.raises(ValueError, match="type"):
        audit([generated(kind="unknown")], [track()])


def test_fma_two_header_rows_and_index_name(tmp_path):
    path = tmp_path / "tracks.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([
            ["", "track", "artist", "artist", "set", "track"],
            ["", "title", "name", "id", "subset", "license"],
            ["track_id", "", "", "", "", ""],
            ["1", "Song, part two", "Artist", "7", "small", "Attribution"],
        ])
    rows = read_fma(path)
    assert len(rows) == 1
    assert rows[0]["title"] == "Song, part two"
    assert rows[0]["track_id"] == "1"


def test_range_reader_reads_selected_zip_member_without_reading_audio():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("audio.mp3", b"a" * 200000)
        bundle.writestr("metadata.csv", "name\nexample\n")
    payload = archive.getvalue()
    requests = []

    def fetch(start, end):
        requests.append((start, end))
        return payload[start:end + 1]

    with zipfile.ZipFile(RangeReader(len(payload), fetch)) as bundle:
        assert bundle.read("metadata.csv") == b"name\nexample\n"
    assert sum(end - start + 1 for start, end in requests) < 10000
    assert all(start > 190000 for start, _ in requests)


def test_budget_rejects_oversized_request_before_network(tmp_path):
    budget = TransferBudget(tmp_path / "transfers.json", limit=10)
    with pytest.raises(ValueError, match="budget"):
        budget.fetch("https://example.invalid/archive.zip", 0, 10)


def test_server_ignoring_range_is_rejected_without_reading_body(tmp_path, monkeypatch):
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args):
            pytest.fail("Must not read a full archive response")

    monkeypatch.setattr("dataset_audit.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="range"):
        TransferBudget(tmp_path / "transfers.json").fetch("https://example.invalid/a.zip", 0, 5)


def test_transfer_ledger_persists_actual_bytes(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-2/100"}

    monkeypatch.setattr("dataset_audit.urlopen", lambda *args, **kwargs: Response(b"abc"))
    ledger = tmp_path / "transfers.json"
    budget = TransferBudget(ledger, limit=10)
    assert budget.fetch("https://example.invalid/a.zip", 0, 2) == (b"abc", 100)
    assert TransferBudget(ledger, limit=10).used == 3
