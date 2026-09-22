"""Audit tests use invented metadata only; fixtures are not training data."""

import csv
import io
import zipfile

import pytest

from tools.dataset_audit import RangeReader, TransferBudget, audit, read_fma, summarize
from tools.dataset_audit import build_pilot_review


def raw_track(**changes):
    row = {"track_id": "1", "track_title": "Song", "artist_name": "Artist",
           "artist_id": "7", "track_url": "https://freemusicarchive.org/music/Artist/Song",
           "license_url": "https://creativecommons.org/licenses/by/4.0/",
           "license_title": "Attribution 4.0 International",
           "track_date_created": "2016-01-02 12:00:00", "track_date_recorded": ""}
    return {**row, **changes}


def test_pilot_groups_counterparts_and_retains_evidence_without_claiming_authorship():
    matches = audit([generated(), generated(generator="second")], [track()])
    rows = build_pilot_review(matches, [raw_track()])
    assert len(rows) == 1
    assert rows[0]["track_id"] == "1"
    assert rows[0]["artist_id"] == "7"
    assert rows[0]["metadata_status"] == "metadata_complete"
    assert rows[0]["provenance_status"] == "unverified"
    assert rows[0]["license_url"] == raw_track()["license_url"]
    assert rows[0]["track_date_created"] == raw_track()["track_date_created"]
    assert rows[0]["tta_count"] == 2
    assert "TTA/second/song.mp3" in rows[0]["echoes_paths_json"]


@pytest.mark.parametrize("changes,reason", [
    ({"license_url": ""}, "missing_license_url"),
    ({"track_url": ""}, "missing_source_url"),
    ({"artist_id": "9"}, "artist_id_conflict"),
    ({"track_title": "Different song"}, "reference_name_conflict"),
    ({"license_url": "https://creativecommons.org/licenses/by-nc/4.0/"}, "license_conflict"),
    ({"track_date_created": "", "track_date_recorded": ""}, "missing_dates"),
    ({"license_url": "https://creativecommons.org.example.org/licenses/by/4.0/"}, "unrecognized_license_url"),
    ({"license_url": "https://creativecommons.org/licenses/by/3.0/"}, "license_version_conflict"),
])
def test_pilot_marks_missing_or_conflicting_evidence(changes, reason):
    rows = build_pilot_review(audit([generated()], [track()]), [raw_track(**changes)])
    assert rows[0]["metadata_status"] == "needs_review"
    assert reason in rows[0]["review_reasons"]


def test_legacy_public_domain_link_is_not_mislabeled_as_conflicting():
    item = track()
    item["license"] = "Public Domain"
    raw = raw_track(license_title="Public Domain",
                    license_url="http://creativecommons.org/licenses/publicdomain/")
    row = build_pilot_review(audit([generated()], [item]), [raw])[0]
    assert row["metadata_status"] == "needs_review"
    assert row["review_reasons"] == "legacy_public_domain_url"


@pytest.mark.parametrize("raw_rows,reason", [([], "missing_raw_record"),
                                                     ([raw_track(), raw_track()], "duplicate_raw_id")])
def test_pilot_keeps_candidate_when_raw_record_is_missing_or_ambiguous(raw_rows, reason):
    rows = build_pilot_review(audit([generated()], [track()]), raw_rows)
    assert len(rows) == 1
    assert rows[0]["metadata_status"] == "needs_review"
    assert reason in rows[0]["review_reasons"]


def test_pilot_excludes_ambiguous_ata_nd_and_repeated_paths():
    nd_track = track("2", title="Restricted")
    nd_track["license"] = "Attribution-NoDerivatives 4.0 International"
    echoes = [generated(), generated(), generated(kind="ATA"),
              generated("Restricted - Artist", generator="nd"),
              generated("Ambiguous - Artist", generator="ambiguous")]
    matches = audit(echoes, [track(), nd_track, track("3", title="Ambiguous"),
                            track("4", title="Ambiguous")])
    assert build_pilot_review(matches, [raw_track()]) == []


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

    monkeypatch.setattr("tools.dataset_audit.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="range"):
        TransferBudget(tmp_path / "transfers.json").fetch("https://example.invalid/a.zip", 0, 5)


def test_transfer_ledger_persists_actual_bytes(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 0-2/100"}

    monkeypatch.setattr("tools.dataset_audit.urlopen", lambda *args, **kwargs: Response(b"abc"))
    ledger = tmp_path / "transfers.json"
    budget = TransferBudget(ledger, limit=10)
    assert budget.fetch("https://example.invalid/a.zip", 0, 2) == (b"abc", 100)
    assert TransferBudget(ledger, limit=10).used == 3
