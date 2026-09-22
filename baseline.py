"""First exploratory logistic-regression baseline and local-file inference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_inspection import load_audio
from features import FEATURE_NAMES, extract_features
from preprocessing import preprocess_audio

ROOT=Path(__file__).resolve().parent
DEFAULT_OUTPUT=ROOT/'data/baseline_v1'
CONTRACT_FILES=('audio_inspection.py','preprocessing.py','features.py')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def assign_groups(rows: list[dict]) -> dict[str,str]:
    """Fixed 12/4/4 group assignment; known pilot artists must remain in training."""
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
    if model['feature_names']!=list(FEATURE_NAMES):
        raise ValueError('Saved feature order differs from the current feature contract')
    standardized=(matrix-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale'])
    return expit(standardized@np.asarray(model['coefficients'])+model['intercept'])


def train(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    manifest=ROOT/'data/batch_manifest.csv'
    feature_path=ROOT/'data/quality_diagnostics/features.csv'
    measurement_path=ROOT/'data/quality_diagnostics/measurements.json'
    rows=read_csv(manifest)
    measurements=json.loads(measurement_path.read_text(encoding='utf-8'))
    if measurements['manifest_sha256']!=digest(manifest):
        raise ValueError('Features were computed for another manifest')
    cached=[r for r in read_csv(feature_path) if r['variant']=='baseline']
    by_id={r['candidate_id']:r for r in cached}
    if len(rows)!=80 or len(cached)!=80 or len(by_id)!=80 or set(by_id)!={r['candidate_id'] for r in rows}:
        raise ValueError('Expected one unchanged-preprocessing feature vector for each of 80 records')
    for row in rows:
        if by_id[row['candidate_id']]['expected_label']!=row['expected_label']:
            raise ValueError('Feature label mismatch')
    x=np.asarray([[float(by_id[r['candidate_id']][f]) for f in FEATURE_NAMES] for r in rows])
    y=np.asarray([{'human':0,'ai':1}[r['expected_label']] for r in rows])
    if x.shape!=(80,10) or not np.isfinite(x).all():
        raise ValueError('Expected finite feature matrix (80, 10)')
    groups=assign_groups(rows)
    roles=np.asarray([groups[r['group_id']] for r in rows])
    # These assertions catch leakage and class-empty splits in the actual experiment.
    for key in ('reference_track_id','artist_id'):
        for value in {r[key] for r in rows}:
            if len({groups[r['group_id']] for r in rows if r[key]==value})!=1:
                raise ValueError(f'{key} crosses splits')
    for role in ('train','validation','holdout'):
        if set(y[roles==role])!={0,1}:
            raise ValueError(f'{role} must contain both classes')
    pipeline=make_pipeline(StandardScaler(),LogisticRegression(
        C=1.0,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        pipeline.fit(x[roles=='train'],y[roles=='train'])
    scaler,classifier=pipeline.steps[0][1],pipeline.steps[1][1]
    model={'schema_version':1,'experiment':'baseline_v1','feature_names':list(FEATURE_NAMES),
        'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
        'coefficients':classifier.coef_[0].tolist(),'intercept':float(classifier.intercept_[0]),
        'threshold':0.5,'start_seconds':0.0,'calibrated':False,
        'class_labels':['human','ai'],'sklearn_version':sklearn.__version__,
        'contract_sha256':{f:digest(ROOT/f) for f in CONTRACT_FILES},
        'training':{'C':1.0,'class_weight':'balanced','max_iter':1000,
                    'solver':'lbfgs','iterations':int(classifier.n_iter_[0])}}
    scores=pipeline.predict_proba(x)[:,1]
    np.testing.assert_allclose(score_features(model,x),scores,rtol=1e-12,atol=1e-12)
    # One real-file round trip verifies the cached training and inference paths agree.
    sample=rows[0]
    fresh=extract_features(preprocess_audio(load_audio(ROOT/sample['file_path']),start_seconds=0.0))
    np.testing.assert_allclose(fresh,x[0],rtol=1e-10,atol=1e-10)
    predicted=(scores>=model['threshold']).astype(int)
    metrics={}
    for role in ('train','validation','holdout'):
        mask=roles==role
        matrix=confusion_matrix(y[mask],predicted[mask],labels=[0,1])
        tn,fp,fn,tp=matrix.ravel()
        metrics[role]={'tracks':int(mask.sum()),'groups':len({rows[i]['group_id'] for i in np.flatnonzero(mask)}),
            'human_tracks':int((y[mask]==0).sum()),'ai_tracks':int((y[mask]==1).sum()),
            'ai_precision':float(precision_score(y[mask],predicted[mask])) if tp+fp else None,
            'ai_recall':float(recall_score(y[mask],predicted[mask])),
            'human_false_positive_rate':float(fp/(tn+fp)),
            'accuracy':float(accuracy_score(y[mask],predicted[mask])),
            'confusion_matrix_true_rows_predicted_columns_human_ai':matrix.tolist()}
    output.mkdir(parents=True)
    (output/'model.json').write_text(json.dumps(model,indent=2)+'\n',encoding='utf-8')
    (output/'metrics.json').write_text(json.dumps({'evaluation':'exploratory; batch previously used for diagnostics',
        'threshold_policy':'fixed 0.5; no threshold search or calibration',
        'split_policy':'12/4/4 artist groups; pilot groups forced to train; baseline-v1 hash order',
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (manifest,feature_path,measurement_path)},
        'metrics':metrics},indent=2)+'\n',encoding='utf-8')
    with (output/'predictions.csv').open('x',encoding='utf-8',newline='') as handle:
        fields=['candidate_id','reference','artist_id','group_id','generator','split','true_label','predicted_label','ai_score']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            writer.writerow({k:row[k] for k in fields[:5]}|{'split':roles[i],'true_label':row['expected_label'],
                'predicted_label':['human','ai'][predicted[i]],'ai_score':float(scores[i])})
    print(json.dumps(metrics,indent=2))
    print(f'Saved model and experiment artifacts to {output}')


def predict(path: Path, model_path: Path) -> dict:
    model=json.loads(model_path.read_text(encoding='utf-8'))
    if any(digest(ROOT/name)!=expected for name,expected in model['contract_sha256'].items()):
        raise ValueError('Audio/feature code changed since training; rerun the experiment explicitly')
    vector=extract_features(preprocess_audio(load_audio(path),start_seconds=model['start_seconds']))
    score=float(score_features(model,vector))
    return {'prediction':'likely_ai_generated' if score>=model['threshold'] else 'likely_human_made',
            'ai_score':score,'threshold':model['threshold'],'calibrated':False,
            'scope':'exploratory direct-file baseline; first 10 seconds; no inconclusive rule yet'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    fit=commands.add_parser('train');fit.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    infer=commands.add_parser('predict');infer.add_argument('audio',type=Path)
    infer.add_argument('--model',type=Path,default=DEFAULT_OUTPUT/'model.json')
    args=parser.parse_args()
    if args.command=='train':train(args.output)
    else:print(json.dumps(predict(args.audio,args.model),indent=2))


if __name__=='__main__':
    main()
