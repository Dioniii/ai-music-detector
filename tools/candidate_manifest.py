"""Expand audited metadata into unsplit recording candidates; never access the network.

Archive sizes describe compressed member payloads, not total HTTP transfer costs.
Artist groups remain provisional until duplicate and artist-alias review is complete.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import unicodedata

from tools.dataset_audit import ECHOES_REVISION, SOURCES

FMA_AUDIO_URL = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
ECHOES_PAGE = f'https://huggingface.co/datasets/Octavian97/Echoes/tree/{ECHOES_REVISION}'
ECHOES_LICENSE = 'https://creativecommons.org/licenses/by-sa/4.0/'


def _safe_path(value: str) -> bool:
    parts = value.split('/')
    return bool(value) and not PurePosixPath(value).is_absolute() and not any(
        p in {'', '.', '..'} or ':' in p or '\\' in p for p in parts)


def _index(entries: list[dict]) -> dict[str, dict]:
    result = {}
    for entry in entries:
        name = entry['name']
        if name in result:
            raise ValueError(f'Duplicate archive member: {name}')
        result[name] = entry
    return result


def _normalized(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', value).casefold().split())


def build_candidates(
    reviews: list[dict], echoes: list[dict], fma_index: list[dict],
    echo_index: list[dict], pilot: list[dict],
) -> list[dict]:
    """Join approved-label candidates to archive metadata without admitting audio.

    Review flags describe the human reference and propagate to its counterparts.
    AI licensing and source fields describe Echoes, separately from that reference.
    Reject ambiguous joins rather than choosing the first matching source row.
    """
    if not reviews or not pilot:
        raise ValueError('Candidate review and pilot restriction records are required')
    by_reference = {}
    for review in reviews:
        tid, artist = review['track_id'], review['artist_id']
        if (not tid.isdecimal() or str(int(tid)) != tid or not artist.isdecimal()
                or artist != review['raw_artist_id']
                or review['reference_group_id'] != f'fma:{tid}'
                or review['artist_group_id'] != f'fma_artist:{artist}'):
            raise ValueError('Inconsistent reference or artist identifiers')
        if tid in by_reference:
            raise ValueError(f'Duplicate reference: {tid}')
        if review['echoes_revision'] != ECHOES_REVISION:
            raise ValueError('Review revision does not match pinned Echoes source')
        if review['metadata_status'] not in {'metadata_complete', 'needs_review'}:
            raise ValueError('Unknown metadata review status')
        by_reference[tid] = review
    pilot_artists = set()
    for record in pilot:
        review = by_reference.get(record['reference_track_id'])
        if review is None or any(record[key] != review[key] for key in
                ('artist_id', 'reference_group_id', 'artist_group_id')):
            raise ValueError('Pilot reference/group mismatch')
        pilot_artists.add(record['artist_id'])

    by_path = defaultdict(list)
    for row in echoes:
        by_path[row['path_in_dataset']].append(row)
    indexes = {'fma_small': _index(fma_index), 'echoes': _index(echo_index)}
    used = set()
    result = []

    def add(review: dict, archive: str, member: str, generated: dict | None) -> None:
        if not _safe_path(member) or (archive, member) in used:
            raise ValueError(f'Unsafe or reused candidate path: {member}')
        entry = indexes[archive].get(member)
        if entry is None:
            raise ValueError(f'Missing archive member: {member}')
        size, compressed = entry['bytes'], entry['compressed_bytes']
        if any(type(n) is not int or n <= 0 for n in (size, compressed)):
            raise ValueError(f'Invalid member sizes: {member}')
        used.add((archive, member))
        ai = generated is not None
        flags = ['audio_integrity_content_review_pending', 'duplicate_alias_review_pending',
                 'usage_review_pending']
        if review['metadata_status'] == 'needs_review' or review['review_reasons']:
            flags.append('reference_metadata_review')
        # This is an existing pilot finding, not a newly inferred exclusion rule.
        if not ai and review['track_id'] == '45101':
            flags.append('pilot_waveform_offset_review_pending')
        result.append({
            'candidate_id': f'ai:{member}' if ai else f"human:fma:{review['track_id']}",
            'expected_label': 'ai' if ai else 'human',
            'label_basis': 'Echoes TTA dataset label' if ai else 'Historical FMA reference label',
            'label_standard': 'dataset_labels_v1',
            'reference': review['reference'], 'reference_track_id': review['track_id'],
            'artist_id': review['artist_id'],
            'reference_group_id': review['reference_group_id'],
            'artist_group_id': review['artist_group_id'],
            'group_id': review['artist_group_id'], 'grouping_status': 'provisional_artist_id',
            'development_only': review['artist_id'] in pilot_artists,
            'split': 'unassigned', 'eligibility_status': 'needs_review',
            'review_reasons': ';'.join(flags),
            'reference_metadata_status': review['metadata_status'],
            'reference_review_reasons': review['review_reasons'],
            'reference_provenance_status': review['provenance_status'],
            'reference_source_url': review['source_url'],
            'reference_license_url': review['license_url'],
            'source_url': ECHOES_PAGE if ai else review['source_url'],
            'license_url': ECHOES_LICENSE if ai else review['license_url'],
            'generator': generated['generator'] if ai else '',
            'generation_type': 'TTA' if ai else '',
            'echoes_genre': generated['genre'] if ai else '',
            'echoes_revision': ECHOES_REVISION,
            'archive': archive, 'archive_url': SOURCES['echoes'] if ai else FMA_AUDIO_URL,
            'archive_member': member, 'file_bytes': size, 'compressed_bytes': compressed,
        })

    for tid, review in sorted(by_reference.items(), key=lambda pair: int(pair[0])):
        padded = f'{int(tid):06d}'
        add(review, 'fma_small', f'fma_small/{padded[:3]}/{padded}.mp3', None)
        try:
            paths = json.loads(review['echoes_paths_json'])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError('Invalid generated path list') from error
        if (not isinstance(paths, list) or not paths
                or any(not isinstance(p, str) or not _safe_path(p) for p in paths)
                or len(paths) != int(review['tta_count']) or len(set(paths)) != len(paths)):
            raise ValueError('Invalid generated paths or TTA count')
        for path in sorted(paths):
            matches = by_path.get(path, [])
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous Echoes row: {path}')
            row = matches[0]
            mode = review['match_status']
            reference_matches = (row['original_audio'] == review['reference'] if mode == 'exact'
                else _normalized(row['original_audio']) == _normalized(review['reference'])
                if mode == 'normalized' else False)
            if (row['type'] != 'TTA' or not row['generator']
                    or not path.startswith(f"TTA/{row['generator']}/") or not reference_matches):
                raise ValueError(f'Inconsistent TTA reference/generator: {path}')
            add(review, 'echoes', f'Echoes/{path}', row)
    return result


def write_manifest(path: Path, rows: list[dict]) -> None:
    """Write exclusively so an existing manifest and manual reviews survive reruns."""
    if not rows:
        raise ValueError('Cannot write an empty candidate manifest')
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=Path('data/preparation/pilot_review.csv'))
    parser.add_argument('--pilot', type=Path, default=Path('data/preparation/audio_pilot_manifest.csv'))
    parser.add_argument('--metadata-dir', type=Path, default=Path('data/source_metadata'))
    parser.add_argument('--output', type=Path, default=Path('data/preparation/candidate_manifest.csv'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [args.review, args.metadata_dir / 'echoes_dataset_manifest.csv',
             args.metadata_dir / 'fma_small_archive_index.json',
             args.metadata_dir / 'echoes_archive_index.json', args.pilot]
    data = []
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            data.append(json.load(handle) if path.suffix == '.json' else list(csv.DictReader(handle)))
    rows = build_candidates(*data)
    write_manifest(args.output, rows)
    print(json.dumps({'output': str(args.output), 'candidate_records': len(rows),
        'labels': dict(Counter(r['expected_label'] for r in rows)),
        'provisional_artist_groups': len({r['group_id'] for r in rows}),
        'development_only_records': sum(r['development_only'] for r in rows),
        'compressed_member_bytes': sum(r['compressed_bytes'] for r in rows),
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=2))


if __name__ == '__main__':
    main()
