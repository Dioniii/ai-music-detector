"""Prepare the historical batch or the 1,000-recording training dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from audio_inspection import load_audio
from tools.candidate_manifest import FMA_AUDIO_URL, write_manifest
from tools.dataset_audit import RangeReader, SOURCES, TransferBudget
from tools.pilot_audio import download_member

LIMIT = 200 * 1024**2
PAYLOAD_LIMIT = 180 * 1024**2  # Reserve room for ZIP indexes and request overlap.
GENERATORS = ('acestep', 'audioldm', 'musicgen')
ARCHIVES = {'fma_small': FMA_AUDIO_URL, 'echoes': SOURCES['echoes']}
BATCH = Path('data/audio_batch')
MANIFEST = Path('data/preparation/batch_manifest.csv')
LEDGER = Path('data/source_metadata/batch_audio_transfers.json')


def select_batch(candidates: list[dict]) -> list[dict]:
    """Choose one reference per 20 artist groups, independently of audio/model scores.

    Group order is SHA-256('batch-v1:' + group_id). Within a group choose the
    lowest numeric eligible reference ID. Use one lexicographically first member
    per fixed generator; incomplete generator sets are left for later batches.
    This metadata selection does not confer final audio eligibility.
    """
    refs = defaultdict(list)
    for row in candidates:
        refs[row['reference_track_id']].append(row)
    groups = defaultdict(list)
    for tid, rows in refs.items():
        humans = [r for r in rows if r['expected_label'] == 'human']
        if len(humans) != 1:
            raise ValueError('Each candidate reference must have exactly one human row')
        human = humans[0]
        if (human['reference_metadata_status'] != 'metadata_complete'
                or human['reference_review_reasons']
                or 'pilot_waveform_offset_review_pending' in human['review_reasons']):
            continue
        selected = [human]
        for generator in GENERATORS:
            options = [r for r in rows if r['expected_label'] == 'ai' and r['generator'] == generator]
            if options:
                selected.append(min(options, key=lambda r: r['archive_member']))
        if len(selected) == 4:
            if any(r['group_id'] != human['group_id'] or r['split'] != 'unassigned'
                   or r['reference_metadata_status'] != 'metadata_complete'
                   or r['reference_review_reasons'] for r in selected):
                raise ValueError('Inconsistent reference grouping/review/split')
            groups[human['group_id']].append(selected)
    if len(groups) < 20:
        raise ValueError('Need 20 groups with complete three-generator coverage')
    selected = []
    order = sorted(groups, key=lambda g: hashlib.sha256(('batch-v1:' + g).encode()).hexdigest())
    for group in order[:20]:
        selected.extend(dict(r) for r in min(groups[group], key=lambda rs: int(rs[0]['reference_track_id'])))
    if sum(int(r['compressed_bytes']) for r in selected) > PAYLOAD_LIMIT:
        raise ValueError('Selected payload exceeds 180 MiB planning ceiling')
    return selected


def verify_existing(path: Path, row: dict) -> bool:
    """Reuse only a complete file whose receipt matches source, size and local hash."""
    receipt_path = path.with_suffix('.receipt.json')
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError(f'Missing file or receipt: {path}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('source_url') != row['archive_url']
            or receipt.get('archive_member') != row['archive_member']
            or path.stat().st_size != int(row['file_bytes'])
            or receipt.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError(f'Existing file receipt mismatch: {path}')
    return True


def download(rows: list[dict], ledger: Path = LEDGER) -> None:
    """Download named members only, persisting a 200 MiB response-body budget.

    Fail closed on changed plans, missing receipts, mismatched member sizes or
    range refusal. Completed downloads can be reused on a subsequent invocation.
    """
    for row in rows:
        if row['archive'] not in ARCHIVES or row['archive_url'] != ARCHIVES[row['archive']]:
            raise ValueError('Unapproved archive URL')
    if not rows or len(rows) > 80 or len({r['group_id'] for r in rows}) > 20:
        raise ValueError('Batch exceeds approved scope')
    identities = {(r['archive'], r['archive_member']) for r in rows}
    if len(identities) != len(rows):
        raise ValueError('Repeated selected member')
    for row in rows:
        target = Path(row['file_path']).resolve()
        roots = [BATCH.resolve(), Path('data/audio_pilot').resolve()]
        if not any(target.is_relative_to(root) for root in roots):
            raise ValueError('Audio target outside batch/pilot directory')
    ledger.parent.mkdir(parents=True, exist_ok=True)
    plan = ledger.with_suffix('.plan.json')
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if plan.exists():
        if json.loads(plan.read_text()) != {'manifest_sha256': digest, 'limit': LIMIT}:
            raise ValueError('Batch plan changed; do not reset the transfer ledger')
    else:
        if ledger.exists():
            raise ValueError('Transfer ledger exists without its batch plan')
        plan.write_text(json.dumps({'manifest_sha256': digest, 'limit': LIMIT}), encoding='utf-8')
    budget = TransferBudget(ledger, limit=LIMIT)
    pending = []
    for row in rows:
        if verify_existing(Path(row['file_path']), row):
            print(f"Reused: {row['candidate_id']}", flush=True)
        else:
            pending.append(row)
    for source, url in ARCHIVES.items():
        subset = [r for r in pending if r['archive'] == source]
        if not subset:
            continue
        _, size = budget.fetch(url, 0, 0)
        with zipfile.ZipFile(RangeReader(size, lambda a, b: budget.fetch(url, a, b)[0])) as archive:
            allowed = {r['archive_member'] for r in subset}
            for row in subset:
                path = Path(row['file_path'])
                digest = download_member(archive, row['archive_member'], int(row['file_bytes']),
                    int(row['compressed_bytes']), path, allowed)
                receipt = {'candidate_id': row['candidate_id'], 'source_url': url,
                    'archive_member': row['archive_member'], 'sha256': digest,
                    'file_bytes': path.stat().st_size}
                with path.with_suffix('.receipt.json').open('x', encoding='utf-8') as handle:
                    json.dump(receipt, handle, indent=2)
                print(f"Downloaded: {row['candidate_id']} ({budget.used / 1024**2:.2f} MiB used)", flush=True)
    print(f'Transfer body bytes: {budget.used} / {LIMIT}', flush=True)


def inspect_audio(path: Path) -> dict:
    """Measure whole-file properties without modifying audio or assigning eligibility.

    |channel mean| > .05 is a review trigger, not an exclusion or AI signature.
    Exact decoded duplicates hash float32 PCM together with sample rate and shape.
    """
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        audio = load_audio(path)
    except (OSError, ValueError) as error:
        return {'status': 'decode_error', 'error': str(error), 'quality_flags': ['decode_error']}
    samples = audio.samples
    means = np.mean(samples, axis=0, dtype=np.float64)
    peak = float(np.max(np.abs(samples)))
    flags = []
    if audio.duration_seconds < 10:
        flags.append('shorter_than_10_seconds')
    if not np.any(samples):
        flags.append('all_zero')
    if np.any(np.abs(means) > .05):
        flags.append('channel_offset_review')
    if peak >= 1:
        flags.append('at_or_above_full_scale_review')
    if audio.channels not in (1, 2):
        flags.append('unsupported_channel_count')
    pcm = hashlib.sha256(json.dumps([audio.sample_rate, list(samples.shape)]).encode())
    pcm.update(samples.astype('<f4', copy=False).tobytes())
    return {'status': 'ok', 'sha256': digest, 'pcm_sha256': pcm.hexdigest(),
        'sample_rate': audio.sample_rate, 'shape': list(samples.shape),
        'duration_seconds': audio.duration_seconds, 'channel_mean_amplitude': means.tolist(),
        'peak_amplitude': peak, 'rms_amplitude': float(np.sqrt(np.mean(samples.astype('float64')**2))),
        'finite_samples': True, 'quality_flags': flags, 'listening_status': 'pending'}


def duplicate_groups(results: list[dict], field: str) -> list[list[str]]:
    groups = defaultdict(list)
    for result in results:
        if result.get(field):
            groups[result[field]].append(result['candidate_id'])
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))



EXPANDED = Path('data/dataset_1000.csv')
EXPANDED_AUDIO = Path('data/audio_1000')


def expand_dataset():
    """Select 500 per class using metadata only; keep inspected artists in training."""
    if EXPANDED.exists():
        print(f'Using existing selection: {EXPANDED}')
        return
    def order(value):
        return hashlib.sha256(('scale1000:' + value).encode()).hexdigest()
    candidates = read_csv(Path('data/preparation/candidate_manifest.csv'))
    ai = [dict(r) for r in candidates if r['expected_label'] == 'ai'
          and r['reference_metadata_status'] == 'metadata_complete'
          and not r['reference_review_reasons'] and int(r['file_bytes']) <= 8 * 1024**2]
    old = read_csv(MANIFEST) + read_csv(Path('data/preparation/audio_pilot_manifest.csv'))
    seen_groups = {r.get('group_id') or r['artist_group_id'] for r in old}
    fresh = sorted({r['group_id'] for r in ai} - seen_groups, key=order)
    group_roles = {g: 'train' for g in seen_groups}
    # Reserve at least seven new reference-artist groups for each evaluation split.
    for role in ('holdout', 'validation'):
        selected_groups = []
        while len(selected_groups) < 7 or sum(r['group_id'] in selected_groups for r in ai) < 100:
            if not fresh:
                raise ValueError('Insufficient fresh groups for evaluation')
            group = fresh.pop(0)
            selected_groups.append(group)
            group_roles[group] = role
    for row in ai:
        row['split'] = group_roles.setdefault(row['group_id'], 'train')
        row['genre'] = row['echoes_genre']
    selected_ai = []
    for role, count in [('train', 350), ('validation', 75), ('holdout', 75)]:
        pool = sorted([r for r in ai if r['split'] == role], key=lambda r: order(r['candidate_id']))
        genres, generators, artists = Counter(), Counter(), Counter()
        for _ in range(count):
            if not pool:
                raise ValueError(f'Insufficient AI candidates in {role}')
            row = min(pool, key=lambda r: (genres[r['genre']], generators[r['generator']], artists[r['group_id']], order(r['candidate_id'])))
            pool.remove(row); selected_ai.append(row)
            genres[row['genre']] += 1; generators[row['generator']] += 1; artists[row['group_id']] += 1
    # Read the public FMA metadata, using only files present in the small archive.
    index = {r['name']: r for r in json.loads(Path('data/source_metadata/fma_small_archive_index.json').read_text())}
    metadata = {}
    with Path('data/source_metadata/fma_tracks.csv').open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle); upper = next(reader); lower = next(reader); next(reader)
        genre_col = list(zip(upper, lower)).index(('track', 'genre_top'))
        for row in reader:
            member = f'fma_small/{int(row[0]) // 1000:03d}/{int(row[0]):06d}.mp3'
            if member in index:
                metadata[row[0]] = (row[genre_col], member)
    humans = []
    with Path('data/source_metadata/fma_raw_tracks.csv').open(encoding='utf-8-sig', newline='') as handle:
        for source in csv.DictReader(handle):
            tid = source['track_id']
            if tid not in metadata or not source['artist_id']:
                continue
            genre, member = metadata[tid]
            license_url = source['license_url']
            if (genre not in {'Pop', 'Rock', 'Electronic'}
                    or 'creativecommons.org/licenses/by' not in license_url or '-nd' in license_url):
                continue
            group = 'fma_artist:' + source['artist_id']
            if group not in group_roles:
                bucket = int(order(group), 16) % 100
                group_roles[group] = 'train' if bucket < 70 else 'validation' if bucket < 85 else 'holdout'
            entry = index[member]
            humans.append({'candidate_id': 'human:fma:' + tid, 'expected_label': 'human',
                'reference': source['track_title'] + ' - ' + source['artist_name'],
                'reference_track_id': tid, 'artist_id': source['artist_id'], 'group_id': group,
                'generator': '', 'genre': genre, 'split': group_roles[group],
                'development_only': group in seen_groups, 'label_basis': 'Historical FMA reference label',
                'archive': 'fma_small', 'archive_url': FMA_AUDIO_URL, 'archive_member': member,
                'file_bytes': entry['bytes'], 'compressed_bytes': entry['compressed_bytes'],
                'source_url': source['track_url'], 'license_url': license_url})
    selected_human = []
    for role in ('train', 'validation', 'holdout'):
        target = Counter(r['genre'] for r in selected_ai if r['split'] == role)
        artist_counts = Counter()
        for genre, count in sorted(target.items()):
            pool = [r for r in humans if r['split'] == role and r['genre'] == genre]
            if len(pool) < count:
                raise ValueError(f'Insufficient human {genre} in {role}: {len(pool)} < {count}')
            for _ in range(count):
                row = min(pool, key=lambda r: (artist_counts[r['group_id']], order(r['candidate_id'])))
                pool.remove(row); selected_human.append(row); artist_counts[row['group_id']] += 1
    rows = selected_human + selected_ai
    previous = {(r['archive'], r['archive_member']): r for r in old}
    for row in rows:
        previous_row = previous.get((row['archive'], row['archive_member']))
        row['file_path'] = (previous_row['file_path'] if previous_row and Path(previous_row['file_path']).is_file()
                            else (EXPANDED_AUDIO / (hashlib.sha256(row['candidate_id'].encode()).hexdigest()
                                                  + Path(row['archive_member']).suffix)).as_posix())
        row['development_only'] = row['group_id'] in seen_groups
    fields = ['candidate_id', 'expected_label', 'reference', 'reference_track_id', 'artist_id',
              'group_id', 'generator', 'genre', 'split', 'development_only', 'label_basis',
              'archive', 'archive_url', 'archive_member', 'file_bytes', 'compressed_bytes',
              'source_url', 'license_url', 'file_path']
    with EXPANDED.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    summary = {'records': len(rows), 'compressed_payload_bytes': sum(int(r['compressed_bytes']) for r in rows),
               'labels': dict(Counter(r['expected_label'] for r in rows)),
               'generators': dict(Counter(r['generator'] for r in selected_ai)),
               'split_counts': {role: dict(Counter(r['expected_label'] for r in rows if r['split'] == role))
                                for role in ('train', 'validation', 'holdout')},
               'groups_per_split': {role: len({r['group_id'] for r in rows if r['split'] == role})
                                   for role in ('train', 'validation', 'holdout')},
               'inspected_groups_training_only': all(r['split'] == 'train' for r in rows if r['group_id'] in seen_groups),
               'limitations': 'Dataset labels are assumptions; artist IDs are provisional; no near-duplicate or listening review.'}
    Path('data/dataset_1000_plan.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def download_expanded():
    """Fetch only selected ZIP members with resumable receipts and a 4 GiB cap."""
    rows = read_csv(EXPANDED)
    if len(rows) != 1000 or Counter(r['expected_label'] for r in rows) != {'human': 500, 'ai': 500}:
        raise ValueError('Expected the approved 500+500 selection')
    budget = TransferBudget(Path('data/source_metadata/expanded_transfers.json'), limit=4 * 1024**3)
    plan_path = Path('data/source_metadata/expanded_transfers.plan.json')
    plan_hash = hashlib.sha256(EXPANDED.read_bytes()).hexdigest()
    if plan_path.exists() and json.loads(plan_path.read_text())['manifest_sha256'] != plan_hash:
        raise ValueError('Selection changed after download started')
    plan_path.write_text(json.dumps({'manifest_sha256': plan_hash, 'limit': budget.limit}))
    completed = 0
    for source, url in ARCHIVES.items():
        pending = []
        for row in [r for r in rows if r['archive'] == source]:
            target = Path(row['file_path'])
            if row['archive_url'] != url or not any(target.resolve().is_relative_to(p.resolve())
                    for p in [EXPANDED_AUDIO, BATCH, Path('data/audio_pilot')]):
                raise ValueError('Unexpected download source or target')
            if verify_existing(target, row):
                completed += 1
            else:
                pending.append(row)
        if not pending:
            continue
        def request_range(start, end):
            for attempt in range(3):
                try:
                    return budget.fetch(url, start, end)
                except TimeoutError:
                    if attempt == 2:
                        raise
                    print(f'Retrying timed-out range from {source}', flush=True)
        _, size = request_range(0, 0)
        cache_start, cache_data = -1, b''
        def fetch(start, end):
            if cache_start <= start and end < cache_start + len(cache_data):
                return cache_data[start-cache_start:end-cache_start+1]
            return request_range(start, end)[0]
        with zipfile.ZipFile(RangeReader(size, fetch)) as archive:
            offsets = sorted([info.header_offset for info in archive.infolist()] + [archive.start_dir])
            next_offset = dict(zip(offsets[:-1], offsets[1:]))
            allowed = {r['archive_member'] for r in pending}
            for row in pending:
                info = archive.getinfo(row['archive_member'])
                cache_start = info.header_offset
                cache_data = request_range(cache_start, next_offset[cache_start]-1)[0]
                path = Path(row['file_path'])
                checksum = download_member(archive, row['archive_member'], int(row['file_bytes']),
                                           int(row['compressed_bytes']), path, allowed)
                path.with_suffix('.receipt.json').write_text(json.dumps({
                    'candidate_id': row['candidate_id'], 'source_url': url,
                    'archive_member': row['archive_member'], 'sha256': checksum,
                    'file_bytes': path.stat().st_size}, indent=2))
                completed += 1
                if completed % 10 == 0:
                    print(f'Downloaded/reused {completed}/1000 | {budget.used/1024**2:.1f} MiB transferred', flush=True)
    print(f'Complete: {completed}/1000, {budget.used/1024**2:.1f} MiB transferred', flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'download', 'inspect', 'expand', 'download-expanded'])
    args = parser.parse_args()
    if args.command == 'expand':
        expand_dataset()
        return
    if args.command == 'download-expanded':
        download_expanded()
        return
    if args.command == 'select':
        rows = select_batch(read_csv(Path('data/preparation/candidate_manifest.csv')))
        pilot = {(r['archive'], r['archive_member']): r for r in read_csv(Path('data/preparation/audio_pilot_manifest.csv'))}
        for row in rows:
            original = pilot.get((row['archive'], row['archive_member']))
            if original and verify_existing(Path(original['file_path']), row):
                path = Path(original['file_path'])
            else:
                name = hashlib.sha256(row['candidate_id'].encode()).hexdigest()
                path = BATCH / (name + Path(row['archive_member']).suffix)
            row['file_path'] = path.as_posix()
        write_manifest(MANIFEST, rows)
        print(json.dumps({'records': len(rows), 'payload_bytes': sum(int(r['compressed_bytes']) for r in rows),
            'labels': dict(Counter(r['expected_label'] for r in rows))}, indent=2))
        return
    rows = read_csv(MANIFEST)
    # Reject alterations to selected metadata before any network operation.
    expected = select_batch(read_csv(Path('data/preparation/candidate_manifest.csv')))
    if [{k: v for k, v in r.items() if k != 'file_path'} for r in rows] != expected:
        raise ValueError('Batch no longer matches deterministic candidate selection')
    if args.command == 'download':
        download(rows)
    else:
        results = []
        for row in rows:
            path = Path(row['file_path'])
            try:
                if not verify_existing(path, row):
                    raise ValueError('Audio not downloaded')
                result = inspect_audio(path)
            except (OSError, ValueError) as error:
                result = {'status': 'integrity_error', 'error': str(error), 'quality_flags': ['integrity_error']}
            results.append({'candidate_id': row['candidate_id'], 'expected_label': row['expected_label'],
                'group_id': row['group_id'], 'file_path': row['file_path'], **result})
            print(f"Checked {len(results)}/{len(rows)}: {result['status']}", flush=True)
        report = {'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'results': results, 'exact_file_duplicates': duplicate_groups(results, 'sha256'),
            'exact_pcm_duplicates': duplicate_groups(results, 'pcm_sha256'),
            'near_duplicate_review': 'pending'}
        with Path('data/preparation/batch_audio_results.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({'statuses': dict(Counter(r['status'] for r in results)),
            'quality_flags': dict(Counter(f for r in results for f in r['quality_flags'])),
            'exact_file_duplicates': report['exact_file_duplicates'],
            'exact_pcm_duplicates': report['exact_pcm_duplicates']}, indent=2))


if __name__ == '__main__':
    main()
