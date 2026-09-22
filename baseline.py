"""Train and predict with the current random-section model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from audio_inspection import load_audio
from features import FEATURE_NAMES, ENCODER_NAMES, ENCODER_DIR, recording_features, encoder_features
from preprocessing import SAMPLING_SEED

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'data/baseline_encoder'
MODEL_PATH = DEFAULT_OUTPUT / 'model.json'
REFERENCE_MODEL_PATH = ROOT / 'data/baseline_random/demo/model.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def assign_groups(rows: list[dict]) -> dict[str,str]:
    # Expanded manifests carry their split so it stays fixed across training runs.
    if any(row.get('split') in {'train', 'validation', 'holdout'} for row in rows):
        groups = {}
        artists = {}
        references = {}
        for row in rows:
            role = row.get('split')
            if role not in {'train', 'validation', 'holdout'}:
                raise ValueError('Every recording must have a split')
            for mapping, key in [(groups, row['group_id']), (artists, row['artist_id']),
                                 (references, row['reference_track_id'])]:
                if mapping.setdefault(key, role) != role:
                    raise ValueError('Related artist/reference recordings cross splits')
            if row.get('development_only') == 'True' and role != 'train':
                raise ValueError('Previously inspected recordings must stay in training')
        return groups
    # Historical 80-recording manifests keep their original 12/4/4 group split.
    groups={r['group_id'] for r in rows}
    forced={r['group_id'] for r in rows if r['development_only']=='True'}
    if len(groups)!=20 or len(forced)>12:
        raise ValueError('Expected 20 groups with at most 12 pilot-restricted groups')
    remaining=sorted(groups-forced,key=lambda g:hashlib.sha256(('baseline-v1:'+g).encode()).hexdigest())
    training=sorted(forced)+remaining[:12-len(forced)]
    rest=remaining[12-len(forced):]
    return {**{g:'train' for g in training},**{g:'validation' for g in rest[:4]},
            **{g:'holdout' for g in rest[4:]}}


def score_features(model: dict, matrix: np.ndarray) -> np.ndarray:
    """Reproduce the saved binary pipeline using portable numeric JSON parameters."""
    names = ENCODER_NAMES if model.get('variant') == 'efficientat' else FEATURE_NAMES
    if model['feature_names']!=list(names):
        raise ValueError('Saved feature order differs from the current feature contract')
    standardized=(matrix-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale'])
    return expit(standardized@np.asarray(model['coefficients'])+model['intercept'])


def load_model(path=MODEL_PATH):
    """Read weights and check that they match the current audio pipeline."""
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    variant = model.get('variant')
    names = ENCODER_NAMES if variant == 'efficientat' else FEATURE_NAMES
    if model['feature_names'] != list(names) or variant not in {'multi_raw', 'efficientat'}:
        raise ValueError('Unknown model feature contract')
    if variant == 'efficientat':
        provenance = json.loads((ENCODER_DIR / 'provenance.json').read_text())
        if model.get('encoder_sha256') != provenance['export_sha256']:
            raise ValueError('Classifier was trained with a different encoder')
    policy = model.get('section_policy', {})
    if (policy.get('sampling') != 'stratified_random' or policy.get('seed') != SAMPLING_SEED
            or policy.get('max_sections') != 5 or policy.get('max_section_seconds') != 20):
        raise ValueError('Saved model uses a different section selection policy')
    if variant == 'efficientat' and policy.get('sample_rate') != 32000:
        raise ValueError('EfficientAT requires the 32 kHz sampling policy')
    return model


def predict(path, model_path=MODEL_PATH):
    model = load_model(model_path)
    extract = encoder_features if model['variant'] == 'efficientat' else recording_features
    vector, sections = extract(load_audio(path))
    standardized = (vector - np.array(model['scaler_mean'])) / np.array(model['scaler_scale'])
    contributions = standardized * np.array(model['coefficients'])
    score = float(score_features(model, vector))
    return {
        'file': Path(path).name, 'ai_score': score,
        'prediction': 'likely_ai_generated' if score >= model['threshold'] else 'likely_human_made',
        'calibrated': False, 'threshold': model['threshold'], 'sections': sections,
        'feature_names': model['feature_names'], 'features': vector.tolist(),
        'contributions': contributions.tolist(), 'intercept': model['intercept'],
    }


def measure(labels, scores, group_count):
    matrix = confusion_matrix(labels, scores >= 0.5, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        'tracks': int(len(labels)), 'groups': group_count,
        'human_tracks': int(tn + fp), 'ai_tracks': int(fn + tp),
        'confusion_matrix_true_rows_predicted_columns_human_ai': matrix.tolist(),
        'ai_precision': float(tp / (tp + fp)) if tp + fp else 0.0,
        'ai_recall': float(tp / (tp + fn)),
        'human_false_positive_rate': float(fp / (tn + fp)),
        'accuracy': float((tn + tp) / len(labels)),
    }


def train(output, manifest=ROOT / "data/dataset_1000.csv", encoder=False):
    """Fit one model on training artists, then evaluate the fixed splits."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a new folder')
    rows = read_csv(Path(manifest))
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated recording ID in manifest')
    groups = assign_groups(rows)
    roles = np.array([groups[row['group_id']] for row in rows])
    labels = np.array([row['expected_label'] == 'ai' for row in rows], dtype=int)
    names = ENCODER_NAMES if encoder else FEATURE_NAMES
    vectors, section_log, comparison_vectors = [], [], []
    encoder_hash = None
    if encoder:
        encoder_hash = json.loads((ENCODER_DIR / 'provenance.json').read_text())['export_sha256']
        cache_dir = ROOT / 'data/embedding_cache'
        cache_dir.mkdir(exist_ok=True)
        cache_version = encoder_hash + digest(ROOT / 'features.py') + digest(ROOT / 'preprocessing.py')
        # Reuse measured baseline features only when the decoded audio matches.
        raw_dir = ROOT / 'data/baseline_1000'
        raw_features = {r['candidate_id']: r for r in read_csv(raw_dir / 'features.csv')} if (raw_dir / 'features.csv').exists() else {}
        raw_hashes = {r['candidate_id']: r['pcm_sha256'] for r in json.loads((raw_dir / 'sections.json').read_text())} if raw_features else {}
    decoded_hashes = {}
    for index, row in enumerate(rows, 1):
        audio = load_audio(ROOT / row['file_path'])
        pcm = hashlib.sha256(json.dumps([audio.sample_rate, list(audio.samples.shape)]).encode())
        pcm.update(audio.samples.astype('<f4', copy=False).tobytes())
        if pcm.hexdigest() in decoded_hashes:
            raise ValueError(f"Duplicate decoded audio: {row['candidate_id']} and {decoded_hashes[pcm.hexdigest()]}")
        decoded_hashes[pcm.hexdigest()] = row['candidate_id']
        if encoder:
            cache_key = hashlib.sha256((pcm.hexdigest() + cache_version).encode()).hexdigest()
            cache_path = cache_dir / (cache_key + '.json')
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                vector, sections = np.array(cached['vector']), cached['sections']
            else:
                vector, sections = encoder_features(audio)
                cache_path.write_text(json.dumps({'vector': vector.tolist(), 'sections': sections}))
            if raw_hashes.get(row['candidate_id']) == pcm.hexdigest():
                comparison_vectors.append([float(raw_features[row['candidate_id']][name]) for name in FEATURE_NAMES])
            else:
                comparison_vectors.append(recording_features(audio)[0])
        else:
            vector, sections = recording_features(audio)
        vectors.append(vector)
        section_log.append({'candidate_id': row['candidate_id'], 'sections': sections,
                            'duration_seconds': audio.duration_seconds, 'pcm_sha256': pcm.hexdigest()})
        if index % 10 == 0:
            print(f'Measured {index}/{len(rows)} recordings', flush=True)
    matrix = np.asarray(vectors)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
    for role in ('train', 'validation', 'holdout'):
        if set(labels[roles == role]) != {0, 1}:
            raise ValueError(f'Missing a class in {role}')
    pipeline.fit(matrix[roles == 'train'], labels[roles == 'train'])
    scaler, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    scores = pipeline.predict_proba(matrix)[:, 1]
    model = {
        'training_manifest': Path(manifest).resolve().relative_to(ROOT).as_posix(),
        'training_manifest_sha256': digest(Path(manifest)),
        'schema_version': 4, 'variant': 'efficientat' if encoder else 'multi_raw', 'feature_names': list(names),
        'encoder_sha256': encoder_hash,
        'threshold': 0.5, 'calibrated': False,
        'section_policy': {'sampling': 'stratified_random', 'seed': SAMPLING_SEED,
                           'rng': 'PCG64', 'sample_rate': 32000 if encoder else 24000, 'max_sections': 5, 'max_section_seconds': 20,
                           'min_recording_seconds': 10, 'aggregation': 'mean of section feature vectors'},
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'coefficients': classifier.coef_[0].tolist(), 'intercept': float(classifier.intercept_[0]),
    }
    metrics = {role: measure(labels[roles == role], scores[roles == role],
                            len({r['group_id'] for r, split in zip(rows, roles) if split == role}))
               for role in ('train', 'validation', 'holdout')}
    previous_model = load_model(REFERENCE_MODEL_PATH)
    previous_scores = score_features(previous_model, np.asarray(comparison_vectors) if encoder else matrix)
    comparison = {'baseline_model_path': REFERENCE_MODEL_PATH.relative_to(ROOT).as_posix(),
                  'baseline_model_sha256': digest(REFERENCE_MODEL_PATH),
                  'previous_model': {role: measure(labels[roles == role], previous_scores[roles == role], metrics[role]['groups'])
                                     for role in ('train', 'validation', 'holdout')},
                  'new_model': metrics, 'exact_pcm_duplicates': 0}
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
    if encoder:
        np.savez_compressed(output / 'features.npz', vectors=matrix,
                            candidate_ids=np.array([r['candidate_id'] for r in rows]))
    else:
        with (output / 'features.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['candidate_id', 'split', 'expected_label', *names])
            for row, role, vector in zip(rows, roles, matrix):
                writer.writerow([row['candidate_id'], role, row['expected_label'], *vector])
    for name, value in [('model.json', model), ('metrics.json', {'metrics': metrics}),
                        ('sections.json', section_log)]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    fields = ['candidate_id', 'reference', 'artist_id', 'group_id', 'generator',
              'split', 'true_label', 'predicted_label', 'ai_score']
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, role, score in zip(rows, roles, scores):
            writer.writerow({key: row[key] for key in fields[:5]} | {
                'split': role, 'true_label': row['expected_label'],
                'predicted_label': 'ai' if score >= 0.5 else 'human', 'ai_score': float(score),
            })
    print(json.dumps(metrics, indent=2))
    print(f'Saved model and evaluation to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train')
    fit.add_argument('--output', type=Path, required=True)
    fit.add_argument('--encoder', action='store_true', help='Train on frozen EfficientAT embeddings')
    fit.add_argument('--manifest', type=Path, default=ROOT / 'data/dataset_1000.csv')
    infer = commands.add_parser('predict')
    infer.add_argument('audio', type=Path)
    infer.add_argument('--model', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.output, args.manifest, encoder=args.encoder)
    else:
        print(json.dumps(predict(args.audio, args.model), indent=2))


if __name__ == '__main__':
    main()
