import copy
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from tools.batch_audio import select_batch, verify_existing, inspect_audio, duplicate_groups, download
from tools.candidate_manifest import FMA_AUDIO_URL


def candidates():
    result=[]
    for n in range(1,26):
        for label,g in [('human',''),('ai','acestep'),('ai','audioldm'),('ai','musicgen')]:
            result.append(dict(candidate_id=f'{n}:{label}:{g}',expected_label=label,generator=g,
                reference_track_id=str(n),group_id=f'fma_artist:{n}',
                reference_metadata_status='metadata_complete',reference_review_reasons='',
                review_reasons='usage_review_pending',development_only=str(n==1),split='unassigned',
                archive_member=f'{n}/{g or "human"}.mp3',compressed_bytes='100',file_bytes='120'))
    return result


def test_selection_is_deterministic_balanced_by_reference_and_preserves_flags():
    source=candidates(); before=copy.deepcopy(source)
    selected=select_batch(source)
    assert len(selected)==80
    assert select_batch(list(reversed(source)))==selected
    assert source==before
    assert len({r['group_id'] for r in selected})==20
    assert sum(r['expected_label']=='human' for r in selected)==20
    assert all(r['split']=='unassigned' and r['review_reasons']=='usage_review_pending' for r in selected)
    assert all(sum(s['reference_track_id']==r['reference_track_id'] for s in selected)==4 for r in selected)


def test_flagged_references_not_downloaded_and_insufficient_pool_fails():
    source=candidates()
    for row in source:
        if int(row['reference_track_id'])<=6:row['reference_metadata_status']='needs_review'
    with pytest.raises(ValueError,match='20'):select_batch(source)


def test_estimated_payload_cannot_consume_transfer_headroom():
    source=candidates()
    for row in source:row['compressed_bytes']=str(3*1024**2)
    with pytest.raises(ValueError,match='payload'):select_batch(source)


def test_existing_receipt_requires_matching_identity_size_and_hash(tmp_path):
    path=tmp_path/'audio.mp3';path.write_bytes(b'abc')
    row={'archive_url':FMA_AUDIO_URL,'archive_member':'fma_small/a.mp3','file_bytes':'3'}
    receipt={'source_url':FMA_AUDIO_URL,'archive_member':row['archive_member'],
             'sha256':hashlib.sha256(b'abc').hexdigest(),'file_bytes':3}
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
    assert verify_existing(path,row)
    path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='receipt'):verify_existing(path,row)


def test_unreceipted_file_is_never_reused(tmp_path):
    path=tmp_path/'audio.mp3';path.write_bytes(b'abc')
    with pytest.raises(ValueError,match='receipt'):verify_existing(path,{})


def test_inspection_flags_silence_shortness_offset_and_corruption(tmp_path):
    silent=tmp_path/'silent.wav';sf.write(silent,np.zeros(8000),8000)
    result=inspect_audio(silent)
    assert {'all_zero','shorter_than_10_seconds'}<=set(result['quality_flags'])
    offset=tmp_path/'offset.wav';sf.write(offset,np.full(80000,.2),8000,subtype='FLOAT')
    assert 'channel_offset_review' in inspect_audio(offset)['quality_flags']
    broken=tmp_path/'bad.wav';broken.write_bytes(b'no audio')
    assert inspect_audio(broken)['status']=='decode_error'


def test_pcm_duplicates_across_encodings_are_found(tmp_path):
    signal=np.sin(np.arange(80000)/10).astype('float32')*.25
    a=tmp_path/'a.wav';b=tmp_path/'b.wav'
    sf.write(a,signal,8000,subtype='PCM_16');sf.write(b,signal,8000,subtype='PCM_24')
    # Use byte-identical decoded values in two distinct container encodings.
    decoded,_=sf.read(a,dtype='float32');sf.write(b,decoded,8000,subtype='FLOAT')
    results=[{'candidate_id':name,**inspect_audio(path)} for name,path in [('a',a),('b',b)]]
    assert results[0]['sha256']!=results[1]['sha256']
    assert duplicate_groups(results,'pcm_sha256')==[['a','b']]


def test_download_rejects_untrusted_archive_before_network(tmp_path):
    row={'archive':'echoes','archive_url':'https://example.org/unapproved.zip'}
    with pytest.raises(ValueError,match='archive'):download([row],tmp_path/'ledger.json')


def test_receipted_batch_resume_uses_no_network(tmp_path, monkeypatch):
    from tools import batch_audio
    monkeypatch.setattr(batch_audio, 'BATCH', tmp_path)
    path=tmp_path/'audio.mp3';path.write_bytes(b'abc')
    row={'candidate_id':'human:1','archive':'fma_small','archive_url':FMA_AUDIO_URL,
         'archive_member':'fma_small/a.mp3','file_bytes':'3','compressed_bytes':'3',
         'group_id':'artist:1','file_path':str(path)}
    receipt={'source_url':FMA_AUDIO_URL,'archive_member':row['archive_member'],
             'sha256':hashlib.sha256(b'abc').hexdigest(),'file_bytes':3}
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
    def unexpected_network(*args):
        pytest.fail('Complete receipt-checked batch must not access the network')
    monkeypatch.setattr(batch_audio.TransferBudget,'fetch',unexpected_network)
    ledger=tmp_path/'ledger.json'
    download([row],ledger)
    download([row],ledger)
    assert not ledger.exists()
    changed=dict(row,candidate_id='changed')
    with pytest.raises(ValueError,match='plan changed'):download([changed],ledger)
