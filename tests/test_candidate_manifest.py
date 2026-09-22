"""Behavioral checks for a metadata-only, unsplit candidate manifest."""
import copy
import csv
import json

import pytest

from tools.candidate_manifest import build_candidates, write_manifest
from tools.dataset_audit import ECHOES_REVISION


@pytest.fixture
def inputs():
    reviews = []
    echoes = []
    fma_index = []
    echo_index = []
    for tid, artist in [(1, 10), (2, 10), (3, 20)]:
        path = f'TTA/suno/song_{tid}.mp3'
        reviews.append(dict(track_id=str(tid), reference=f'Song {tid} - Artist {artist}',
            artist_id=str(artist), raw_artist_id=str(artist),
            reference_group_id=f'fma:{tid}', artist_group_id=f'fma_artist:{artist}',
            match_status='exact', source_url=f'https://example.org/{tid}',
            license_url='https://creativecommons.org/licenses/by/4.0/',
            metadata_status='metadata_complete', review_reasons='',
            provenance_status='unverified', echoes_revision=ECHOES_REVISION,
            tta_count='1', echoes_paths_json=json.dumps([path])))
        echoes.append(dict(path_in_dataset=path, original_audio=reviews[-1]['reference'],
            generator='suno', type='TTA', genre='Pop'))
        fma_index.append(dict(name=f'fma_small/000/{tid:06d}.mp3', bytes=100, compressed_bytes=90))
        echo_index.append(dict(name=f'Echoes/{path}', bytes=200, compressed_bytes=180))
    pilot = [dict(reference_track_id='1', artist_id='10', reference_group_id='fma:1',
                  artist_group_id='fma_artist:10')]
    return reviews, echoes, fma_index, echo_index, pilot


def test_labels_sizes_groups_and_no_automatic_admission(inputs):
    rows = build_candidates(*inputs)
    assert len(rows) == 6
    assert [r['expected_label'] for r in rows].count('human') == 3
    assert [r['expected_label'] for r in rows].count('ai') == 3
    for row in rows:
        human = row['expected_label'] == 'human'
        assert row['file_bytes'] == (100 if human else 200)
        assert row['compressed_bytes'] == (90 if human else 180)
        assert row['reference_provenance_status'] == 'unverified'
        assert row['eligibility_status'] == 'needs_review'
        assert row['split'] == 'unassigned'
        assert row['grouping_status'] == 'provisional_artist_id'
        assert row['development_only'] == (row['artist_id'] == '10')
        assert row['label_basis']
        assert row['source_url'] and row['license_url'] and row['archive_url']
    assert len({r['candidate_id'] for r in rows}) == 6
    assert len({(r['archive'], r['archive_member']) for r in rows}) == 6


def test_flags_preserved_and_reference_license_not_assigned_to_ai(inputs):
    inputs[0][0].update(metadata_status='needs_review', review_reasons='legacy_public_domain_url')
    rows = [r for r in build_candidates(*inputs) if r['reference_track_id'] == '1']
    assert len(rows) == 2
    for row in rows:
        assert row['reference_review_reasons'] == 'legacy_public_domain_url'
        assert 'reference_metadata_review' in row['review_reasons']
    ai = next(r for r in rows if r['expected_label'] == 'ai')
    assert ai['license_url'] == 'https://creativecommons.org/licenses/by-sa/4.0/'
    assert ai['reference_license_url'] == inputs[0][0]['license_url']


def test_order_independent_and_does_not_mutate_inputs(inputs):
    original = copy.deepcopy(inputs)
    expected = build_candidates(*inputs)
    assert build_candidates(*(list(reversed(x)) for x in inputs)) == expected
    assert inputs == original


@pytest.mark.parametrize('defect', ['duplicate_reference', 'duplicate_path', 'missing_member',
    'duplicate_member', 'ata', 'reference_mismatch', 'artist_mismatch', 'revision',
    'count', 'unsafe_path', 'negative_size', 'missing_artist', 'pilot_mismatch'])
def test_rejects_inconsistent_sources(inputs, defect):
    reviews, echoes, fma_index, echo_index, pilot = inputs
    if defect == 'duplicate_reference': reviews.append(copy.deepcopy(reviews[0]))
    elif defect == 'duplicate_path': echoes.append(copy.deepcopy(echoes[0]))
    elif defect == 'missing_member': echo_index.pop(0)
    elif defect == 'duplicate_member': echo_index.append(copy.deepcopy(echo_index[0]))
    elif defect == 'ata': echoes[0]['type'] = 'ATA'
    elif defect == 'reference_mismatch': echoes[0]['original_audio'] = 'Wrong song'
    elif defect == 'artist_mismatch': reviews[0]['raw_artist_id'] = '999'
    elif defect == 'revision': reviews[0]['echoes_revision'] = 'different'
    elif defect == 'count': reviews[0]['tta_count'] = '2'
    elif defect == 'unsafe_path': reviews[0]['echoes_paths_json'] = '["../escape.mp3"]'
    elif defect == 'negative_size': echo_index[0]['compressed_bytes'] = -1
    elif defect == 'missing_artist': reviews[0]['artist_id'] = ''
    elif defect == 'pilot_mismatch': pilot[0]['artist_id'] = '999'
    with pytest.raises(ValueError): build_candidates(*inputs)


def test_write_preserves_existing_review_file(tmp_path, inputs):
    rows = build_candidates(*inputs)
    path = tmp_path / 'candidates.csv'
    write_manifest(path, rows)
    with path.open(encoding='utf-8', newline='') as f:
        assert len(list(csv.DictReader(f))) == 6
    before = path.read_bytes()
    with pytest.raises(FileExistsError): write_manifest(path, rows)
    assert path.read_bytes() == before
