"""Shared analysis and figures for the saved audio-classifier showcase."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import soundfile as sf

from audio_inspection import Audio, load_audio, spectrogram
from baseline import ROOT, DEFAULT_OUTPUT, read_csv, score_features, load_model
from features import recording_features, encoder_features
from preprocessing import prepare_recording, TARGET_SAMPLE_RATE

MODEL_PATH=DEFAULT_OUTPUT/'model.json'
GREEN='#5a8a80'
CORAL='#d4a24e'
INK='#e8e8e3'
BACKGROUND='#0d0d0d'
MUTED='#8a8a85'
BORDER='#2a2a28'
ROLE_LABELS={'train': 'Used for learning', 'validation': 'Earlier check', 'holdout': 'Final check'}
FONT_DIR=ROOT/'assets/fonts'
for font_path in FONT_DIR.glob('*.ttf'):
    font_manager.fontManager.addfont(str(font_path))
matplotlib.rcParams.update({'font.family':'monospace','font.monospace':['IBM Plex Mono'],
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':BORDER,
    'axes.facecolor':BACKGROUND,'xtick.color':MUTED,'ytick.color':MUTED})
STEPPED=ListedColormap([BACKGROUND,BORDER,GREEN,CORAL])
PRETTY=['RMS mean','RMS variation','Zero crossings mean','Zero crossings variation',
        'Spectral center mean','Spectral center variation','Bandwidth mean','Bandwidth variation',
        'Flatness mean','Flatness variation']

EMPTY='<div class="result-card"><div class="eyebrow">Ready when you are</div><h2>Start with a recording</h2><p>Upload a song, choose an example or use your microphone, then select Analyze recording.</p><p class="small-note">10 seconds to 5 minutes, up to 50 MB. We check a few sections from across the recording.</p></div>'


def style_axis(axis):
    axis.spines[['top','right']].set_visible(False)
    axis.tick_params(colors=MUTED,labelsize=9)
    for spine in axis.spines.values():spine.set_color(BORDER);spine.set_linewidth(1)
    axis.set_axisbelow(True)
    axis.grid(False)
    axis.xaxis.grid(color=BORDER,alpha=1,linewidth=.5)


def selection_figure(audio, sections, sample_rate=TARGET_SAMPLE_RATE):
    """Full-track coverage and the first sampled section, not just the intro."""
    mono,_ = prepare_recording(audio, sample_rate=sample_rate)
    fig=Figure(figsize=(9,5.1),layout='constrained',facecolor=BACKGROUND)
    waveform,spectrum=fig.subplots(2,1)
    step=max(1,len(mono)//12000)
    waveform.plot(np.arange(0,len(mono),step)/sample_rate,mono[::step],color=GREEN,linewidth=.5)
    for section in sections:
        waveform.axvspan(section['start_seconds'],section['end_seconds'],facecolor='none',edgecolor=CORAL,linewidth=1)
    waveform.set(title='Audio overview | we checked the outlined sections',xlabel='Time (seconds)',ylabel='Sound level',xlim=(0,audio.duration_seconds))
    section=sections[0];start=round(section['start_seconds']*sample_rate);end=round(section['end_seconds']*sample_rate)
    clip=mono[start:end].astype(np.float32)
    power,freq,times=spectrogram(Audio(clip[:,None],sample_rate))
    db=10*np.log10(np.maximum(power,1e-12));bounds=np.linspace(float(db.min()),float(db.max()) if db.max()>db.min() else float(db.min())+1,5)
    im=spectrum.pcolormesh(times+section['start_seconds'],freq/1000,db,shading='auto',cmap=STEPPED,norm=BoundaryNorm(bounds,STEPPED.N))
    spectrum.set(title='Sound map | first selected section',xlabel='Time in recording (seconds)',ylabel='Low to high sounds (kHz)')
    fig.colorbar(im,ax=spectrum,label='Sound strength (dB)')
    for axis in (waveform,spectrum):style_axis(axis)
    return fig


def contribution_figure(contributions,intercept, names=None):
    if names is None:
        values=np.append(contributions,intercept)
        labels=PRETTY+['Starting point']
    else:
        strongest=np.argsort(np.abs(contributions))[-10:]
        remainder=float(np.sum(contributions)-np.sum(contributions[strongest]))
        values=np.append(contributions[strongest], [remainder, intercept])
        labels=[names[i].replace('embedding_', 'Pattern ') for i in strongest]+['Other patterns combined', 'Starting point']
    order=np.argsort(values)
    fig=Figure(figsize=(9,4.8),layout='constrained',facecolor=BACKGROUND);axis=fig.subplots()
    axis.barh(np.asarray(labels)[order],values[order],color=[CORAL if v>=0 else GREEN for v in values[order]],height=.30)
    axis.axvline(0,color=INK,linewidth=.8)
    axis.set(xlabel='Effect on the result | negative: human | positive: AI',title='What influenced this prediction?')
    style_axis(axis)
    return fig


def analyze(path, display_name=None):
    """Return real model measurements; errors clear previous results instead of leaving stale plots."""
    if not path:
        return '<div class="result-card"><h2>Add a recording first</h2><p>Choose a WAV, MP3, FLAC or OGG file of at least 10 seconds.</p></div>','',None,None,None,[]
    try:
        info=sf.info(path)
        if info.duration<10:
            raise ValueError(f'This recording is {info.duration:.2f} seconds long. Please choose one with at least 10 seconds.')
        if info.duration>300:
            raise ValueError('For this demo, please choose a recording no longer than 5 minutes.')
        if info.frames*info.channels>60_000_000:
            raise ValueError('This recording is too large to decode in the demo. Please upload a shorter excerpt.')
        model=load_model(MODEL_PATH)
        audio=load_audio(path)
        encoded=model['variant']=='efficientat'
        vector,sections=(encoder_features(audio) if encoded else recording_features(audio))
        standardized=(vector-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale'])
        contributions=standardized*np.asarray(model['coefficients'])
        score=float(score_features(model,vector))
        prediction='Likely AI-generated' if score>=model['threshold'] else 'Likely human-made'
        result=f'''<div class="result-card" data-result="{'ai' if score>=model['threshold'] else 'human'}"><div class="eyebrow">Model prediction</div><h2>{prediction}</h2>
        <p>AI score <strong class="score-value">{score:.3f}</strong> · AI label starts at {model['threshold']:.2f}</p>
        <div class="score-track"><span class="score-pin" style="left:calc({score*100:.3f}% - 2px)"></span></div>
        <div class="score-labels"><span>0 · leans human</span><span>0.5</span><span>leans AI · 1</span></div>
        <p class="small-note">This is a prediction, not proof. The score is not a percentage chance of AI. The model picks a label even when the score is near the middle.</p></div>'''
        ranges=', '.join(f"{s['start_seconds']:.1f}-{s['end_seconds']:.1f} s" for s in sections)
        details=f"**{html.escape(display_name or Path(path).name)}** | {audio.duration_seconds:.1f} seconds\n\nSections checked: {ranges}.\n\nWe combine information from these sections into one result. Checking this file again uses the same sections."
        names=model['feature_names'] if encoded else PRETTY
        indices=np.argsort(np.abs(contributions))[-10:][::-1] if encoded else range(10)
        table=[[names[i].replace('embedding_', 'Pattern '),float(vector[i]),float(standardized[i]),float(contributions[i])] for i in indices]
        return result,details,str(path),selection_figure(audio,sections,32000 if encoded else TARGET_SAMPLE_RATE),contribution_figure(contributions,model['intercept'],names if encoded else None),table
    except (ValueError,OSError,RuntimeError) as error:
        return f'<div class="result-card"><h2>Could not analyze this recording</h2><p>{html.escape(str(error))}</p></div>','',None,None,None,[]


def confusion_figure(metrics):
    figure=Figure(figsize=(10,3.5),layout='constrained',facecolor=BACKGROUND)
    axes=figure.subplots(1,3)
    for axis,role in zip(axes,['train','validation','holdout']):
        data=np.asarray(metrics[role]['confusion_matrix_true_rows_predicted_columns_human_ai'])
        axis.imshow(data,cmap=STEPPED,vmin=0,vmax=max(1,data.max()),interpolation='nearest')
        axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Human','AI'],yticklabels=['Human','AI'],
                 xlabel='Model prediction',ylabel='Original label',title=f'{ROLE_LABELS[role]} · {metrics[role]["tracks"]} tracks')
        for i in range(2):
            for j in range(2):axis.text(j,i,str(data[i,j]),ha='center',va='center',fontsize=21,color=BACKGROUND if data[i,j]>data.max()/2 else INK)
    return figure


def distribution_figure(predictions):
    fig=Figure(figsize=(10,3.5),layout='constrained',facecolor=BACKGROUND);axes=fig.subplots(1,3)
    for ax,role in zip(axes,['train','validation','holdout']):
        selected=[r for r in predictions if r['split']==role]
        for label,y,color in [('human',0,GREEN),('ai',1,CORAL)]:
            scores=[float(r['ai_score']) for r in selected if r['true_label']==label]
            ax.scatter(scores,y+np.linspace(-.16,.16,len(scores)),color=color,s=35,alpha=1)
        ax.axvline(.5,color=INK,linestyle='--',linewidth=1)
        ax.set(xlim=(-.03,1.03),ylim=(-.45,1.45),yticks=[0,1],yticklabels=['Human','AI'],xlabel='AI score',title=ROLE_LABELS[role])
        style_axis(ax)
    return fig


def local_examples():
    manifest=ROOT/load_model(MODEL_PATH).get('training_manifest', 'data/preparation/batch_manifest.csv')
    if not manifest.exists():return [],[]
    mapping={r['candidate_id']:r for r in read_csv(manifest)}
    predictions=read_csv(DEFAULT_OUTPUT/'predictions.csv')
    examples=[];labels=[]
    for true,pred,label in [('human','human','Human recording · model got it right'),('ai','ai','AI recording · model got it right'),('human','ai','Human recording · model mistook it for AI')]:
        for r in predictions:
            source=mapping.get(r['candidate_id'])
            if r['true_label']==true and r['predicted_label']==pred and source and (ROOT/source['file_path']).exists():
                examples.append([str(ROOT/source['file_path'])]);labels.append(label+' — '+source['reference']);break
    return examples,labels
