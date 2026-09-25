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


@pytest.mark.parametrize('level', [0.0, 0.0001])
def test_recording_gate_rejects_silence_and_very_weak_audio(level):
    from audio_inspection import recording_quality_issue
    t = np.arange(12000) / 1000
    samples = (level * np.sin(2 * np.pi * 100 * t)).astype(np.float32)[:, None]
    assert 'too quiet' in recording_quality_issue(Audio(samples, 1000))


def test_recording_gate_accepts_clear_audio_after_quiet_intro_without_mutating():
    from audio_inspection import recording_quality_issue
    t = np.arange(10000) / 1000
    music = (0.1 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    samples = np.concatenate([np.zeros(10000, np.float32), music])[:, None]
    before = samples.copy()
    assert recording_quality_issue(Audio(samples, 1000)) is None
    np.testing.assert_array_equal(samples, before)


def test_recording_gate_checks_clipping_per_channel_but_allows_isolated_peaks():
    from audio_inspection import recording_quality_issue
    t = np.arange(10000) / 1000
    samples = np.tile((0.1 * np.sin(2 * np.pi * 100 * t))[:, None], (1, 2)).astype(np.float32)
    samples[0, 0] = 1
    assert recording_quality_issue(Audio(samples, 1000)) is None
    samples[:2000, 0] = 1
    assert 'distorted' in recording_quality_issue(Audio(samples, 1000))


def test_recording_gate_does_not_treat_dc_offset_as_music():
    from audio_inspection import recording_quality_issue
    assert 'too quiet' in recording_quality_issue(Audio(np.full((10000, 1), .2, np.float32), 1000))


def test_rejected_recording_never_loads_model_or_returns_prediction(tmp_path, monkeypatch):
    import soundfile as sf
    import showcase
    path = tmp_path / 'silence.wav'
    sf.write(path, np.zeros(320000, np.float32), 32000)
    def unexpected(*args, **kwargs):
        raise AssertionError('Rejected audio must not reach the model')
    monkeypatch.setattr(showcase, 'load_model', unexpected)
    result = showcase.analyze(path)
    assert 'No AI or human prediction was made' in result[0]
    assert result[1:] == ('', None, None, None, [])


@pytest.mark.parametrize('burst_seconds', [0, 1, 2, 3, 4])
def test_quiet_room_with_brief_sound_is_rejected(burst_seconds):
    from audio_inspection import recording_quality_issue
    rate = 1000
    t = np.arange(18300) / rate
    samples = .0025 * np.sin(2 * np.pi * 100 * t)
    samples[:burst_seconds * rate] *= 5
    assert 'too quiet' in recording_quality_issue(Audio(samples.astype(np.float32)[:, None], rate))


def test_partial_last_second_cannot_count_as_full_second():
    from audio_inspection import recording_quality_issue
    t = np.arange(12300) / 1000
    samples = np.zeros(len(t), np.float32)
    samples[-2300:] = .1 * np.sin(2 * np.pi * 100 * t[-2300:])
    assert 'too quiet' in recording_quality_issue(Audio(samples[:, None], 1000))


@pytest.mark.parametrize('seconds, accepted', [(4, False), (5, True)])
def test_recording_gate_requires_five_seconds_above_minimum(seconds, accepted):
    from audio_inspection import recording_quality_issue
    rate = 1000
    t = np.arange(10000) / rate
    samples = np.zeros(10000, dtype=np.float32)
    samples[:seconds * rate] = .1 * np.sin(2 * np.pi * 100 * t[:seconds * rate])
    issue = recording_quality_issue(Audio(samples[:, None], rate))
    assert (issue is None) == accepted
