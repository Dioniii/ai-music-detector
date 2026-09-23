"""Local paired preprocessing diagnostics; never modify audio or production policy."""
import argparse
import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from audio_inspection import Audio, load_audio
from tools.batch_audio import read_csv, verify_existing
from features import FEATURE_NAMES, extract_features
from preprocessing import preprocess_audio


def _validate(samples: np.ndarray) -> None:
    if (not isinstance(samples,np.ndarray) or samples.ndim!=2 or samples.shape[0]==0
            or samples.shape[1] not in (1,2) or not np.issubdtype(samples.dtype,np.floating)
            or not np.isfinite(samples).all()):
        raise ValueError('Expected nonempty finite floating samples shaped (frames, 1 or 2)')


def remove_channel_means(audio: Audio) -> Audio:
    """Subtract each WHOLE-recording channel mean in float64, return new float32 audio.

    This diagnostic can use samples after the target clip. It is not an adopted
    streaming/ten-second-inference preprocessing rule, nor a general high-pass filter.
    """
    _validate(audio.samples)
    if not isinstance(audio.sample_rate,Integral) or isinstance(audio.sample_rate,bool) or audio.sample_rate<=0:
        raise ValueError('Expected positive integer sample rate')
    centered=audio.samples.astype(np.float64)-audio.samples.mean(axis=0,dtype=np.float64)
    return Audio(centered.astype(np.float32),audio.sample_rate)


def full_scale_stats(samples: np.ndarray) -> dict:
    """Counts refer to decoded sample values, not acoustic loudness or proven clipping."""
    _validate(samples)
    magnitude=np.abs(samples)
    at=magnitude>=1.0
    above=magnitude>1.0
    counts=at.sum(axis=0)
    return {'frames':samples.shape[0], 'channels':samples.shape[1], 'sample_values':samples.size,
        'at_or_above_count':int(at.sum()), 'above_count':int(above.sum()),
        'at_or_above_fraction':float(at.mean()), 'above_fraction':float(above.mean()),
        'per_channel_at_or_above_count':counts.tolist(),
        'per_channel_at_or_above_fraction':(counts/samples.shape[0]).tolist(),
        'frames_any_at_or_above_count':int(at.any(axis=1).sum())}


def compare_features(audio: Audio) -> dict:
    """Same first-ten-second preprocessing/feature extractor for both variants."""
    baseline=preprocess_audio(audio,start_seconds=0.0)
    centered=preprocess_audio(remove_channel_means(audio),start_seconds=0.0)
    return {'baseline':extract_features(baseline).tolist(),
        'channel_mean_removed':extract_features(centered).tolist(),
        'clip_shape':list(baseline.shape)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('data/quality_diagnostics'))
    args=parser.parse_args()
    # Exclusive directory avoids overwriting prior diagnostics or listening notes.
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest=Path('data/preparation/batch_manifest.csv')
    rows=read_csv(manifest)
    prior_path=Path('data/preparation/batch_audio_results.json')
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if prior['manifest_sha256']!=manifest_hash:
        raise ValueError('Batch inspection refers to a different manifest')
    flags={r['candidate_id']:r['quality_flags'] for r in prior['results']}
    if len(rows)!=80 or len({r['candidate_id'] for r in rows})!=80 or set(flags)!={r['candidate_id'] for r in rows}:
        raise ValueError('Expected the inspected 80-recording batch')
    measurements=[];features=[]
    for index,row in enumerate(rows,1):
        path=Path(row['file_path'])
        if not verify_existing(path,row):
            raise ValueError(f'Missing receipt-checked audio: {path}')
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        audio=load_audio(path)
        compared=compare_features(audio)
        measurements.append({'candidate_id':row['candidate_id'],'sha256':before,
            'sample_rate':audio.sample_rate,'shape':list(audio.samples.shape),
            'whole_channel_means':audio.samples.mean(axis=0,dtype=np.float64).tolist(),
            'whole_original':full_scale_stats(audio.samples),
            'first_ten_seconds_original':full_scale_stats(audio.samples[:10*audio.sample_rate])})
        for variant in ('baseline','channel_mean_removed'):
            features.append({'candidate_id':row['candidate_id'],'expected_label':row['expected_label'],
                'generator':row['generator'],'variant':variant,
                **dict(zip(FEATURE_NAMES,compared[variant],strict=True))})
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:
            raise ValueError(f'Original audio changed during diagnostic: {path}')
        print(f'Compared {index}/80',flush=True)
    checklist={r['candidate_id']:'Flagged human recording' for r in rows
               if r['expected_label']=='human' and flags[r['candidate_id']]}
    for generator in sorted({r['generator'] for r in rows if r['generator']}):
        example=min((r for r in rows if r['generator']==generator),key=lambda r:int(r['reference_track_id']))
        checklist[example['candidate_id']]='Generator example selected by lowest reference ID'
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'features.csv').open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(features[0]));writer.writeheader();writer.writerows(features)
    payload={'manifest_sha256':manifest_hash,'inspection_sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        'start_seconds':0.0,'centering_scope':'whole_recording_per_channel_before_existing_preprocessing',
        'production_policy_changed':False,'feature_names':list(FEATURE_NAMES),'measurements':measurements}
    with (args.output_dir/'measurements.json').open('x',encoding='utf-8') as handle:
        json.dump(payload,handle,indent=2)
    with (args.output_dir/'listening_checklist.csv').open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['candidate_id','reference','generator','file_path','reason','quality_flags','listening_status','notes'])
        writer.writeheader()
        for row in rows:
            if row['candidate_id'] in checklist:
                writer.writerow({k:row[k] for k in ['candidate_id','reference','generator','file_path']}|
                    {'reason':checklist[row['candidate_id']],'quality_flags':';'.join(flags[row['candidate_id']]),
                     'listening_status':'pending','notes':''})
    print(f'Saved 160 feature rows, 80 measurements and {len(checklist)} listening entries.')


if __name__=='__main__':
    main()
