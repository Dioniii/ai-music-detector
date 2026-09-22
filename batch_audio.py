"""Select, download and inspect the approved 20-group batch; no training or splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from audio_inspection import load_audio
from candidate_manifest import FMA_AUDIO_URL, write_manifest
from dataset_audit import RangeReader, SOURCES, TransferBudget
from pilot_audio import download_member

LIMIT = 200 * 1024**2
PAYLOAD_LIMIT = 180 * 1024**2  # Reserve room for ZIP indexes and request overlap.
GENERATORS = ('acestep', 'audioldm', 'musicgen')
ARCHIVES = {'fma_small': FMA_AUDIO_URL, 'echoes': SOURCES['echoes']}
BATCH = Path('data/audio_batch')
MANIFEST = Path('data/batch_manifest.csv')
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['select', 'download', 'inspect'])
    args = parser.parse_args()
    if args.command == 'select':
        rows = select_batch(read_csv(Path('data/candidate_manifest.csv')))
        pilot = {(r['archive'], r['archive_member']): r for r in read_csv(Path('data/audio_pilot_manifest.csv'))}
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
    expected = select_batch(read_csv(Path('data/candidate_manifest.csv')))
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
        with Path('data/batch_audio_results.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps({'statuses': dict(Counter(r['status'] for r in results)),
            'quality_flags': dict(Counter(f for r in results for f in r['quality_flags'])),
            'exact_file_duplicates': report['exact_file_duplicates'],
            'exact_pcm_duplicates': report['exact_pcm_duplicates']}, indent=2))


if __name__ == '__main__':
    main()
