import numpy as np
import pytest
from audio_inspection import Audio
from tools.quality_diagnostics import remove_channel_means, full_scale_stats, compare_features
from features import FEATURE_NAMES


def test_means_removed_per_channel_without_mutation_or_clipping():
    samples=np.array([[1.5,-.5],[.5,-1.5]],dtype=np.float32)
    before=samples.copy()
    result=remove_channel_means(Audio(samples,24000))
    np.testing.assert_allclose(result.samples,[[.5,.5],[-.5,-.5]])
    np.testing.assert_array_equal(samples,before)
    assert result.sample_rate==24000
    assert not np.shares_memory(result.samples,samples)


def test_full_scale_counts_have_explicit_sample_and_frame_denominators():
    samples=np.array([[1.,0.],[-1.,1.1],[.5,-1.2],[0.,0.]],np.float32)
    result=full_scale_stats(samples)
    assert result['sample_values']==8 and result['frames']==4
    assert result['at_or_above_count']==4 and result['above_count']==2
    assert result['at_or_above_fraction']==.5
    assert result['frames_any_at_or_above_count']==3
    assert result['per_channel_at_or_above_count']==[2,2]
    assert result['per_channel_at_or_above_fraction']==[.5,.5]


@pytest.mark.parametrize('bad',[np.empty((0,1)),np.ones(4),np.ones((3,3)),np.ones((3,1),dtype=int),np.full((2,1),np.nan)])
def test_invalid_samples_rejected(bad):
    with pytest.raises(ValueError):full_scale_stats(bad)
    with pytest.raises(ValueError):remove_channel_means(Audio(bad,24000))


def test_constant_offset_comparison_and_shapes():
    samples=np.full((240000,1),.25,np.float32)
    result=compare_features(Audio(samples,24000))
    before=dict(zip(FEATURE_NAMES,result['baseline'],strict=True))
    after=dict(zip(FEATURE_NAMES,result['channel_mean_removed'],strict=True))
    assert before['rms_mean']==pytest.approx(.25)
    assert after['rms_mean']==0
    assert result['clip_shape']==[240000]
    assert len(result['baseline'])==len(result['channel_mean_removed'])==10
    assert np.all(samples==.25)


def test_whole_recording_mean_is_explicit_not_first_clip_mean():
    samples=np.concatenate([np.full((240000,1),.25),np.full((240000,1),.75)]).astype(np.float32)
    centered=remove_channel_means(Audio(samples,24000))
    assert centered.samples[0,0]==pytest.approx(-.25)
    np.testing.assert_allclose(centered.samples.mean(axis=0),0,atol=1e-7)
