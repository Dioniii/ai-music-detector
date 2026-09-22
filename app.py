"""Local Gradio showcase for the saved handcrafted-feature baseline."""
import argparse
import html
import json
import os
from pathlib import Path

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('GRADIO_TEMP_DIR',str(Path(__file__).resolve().parent/'.gradio_cache'))
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
import inspect
import numpy as np
import soundfile as sf
import gradio as gr

from audio_inspection import Audio, load_audio, spectrogram
from baseline import ROOT, DEFAULT_OUTPUT, CONTRACT_FILES, digest, read_csv, score_features
from features import FEATURE_NAMES, extract_features
from preprocessing import preprocess_audio, TARGET_SAMPLE_RATE

MODEL_PATH=DEFAULT_OUTPUT/'model.json'
GREEN='#5a8a80'
CORAL='#d4a24e'
INK='#e8e8e3'
BACKGROUND='#0d0d0d'
MUTED='#8a8a85'
BORDER='#2a2a28'
FONT_DIR=ROOT/'assets/fonts'
for font_path in FONT_DIR.glob('*.ttf'):
    font_manager.fontManager.addfont(str(font_path))
matplotlib.rcParams.update({'font.family':'monospace','font.monospace':['IBM Plex Mono'],
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':BORDER,
    'axes.facecolor':BACKGROUND,'xtick.color':MUTED,'ytick.color':MUTED})
gr.set_static_paths(paths=[FONT_DIR])
STEPPED=ListedColormap([BACKGROUND,BORDER,GREEN,CORAL])
PRETTY=['RMS mean','RMS variation','Zero crossings mean','Zero crossings variation',
        'Spectral center mean','Spectral center variation','Bandwidth mean','Bandwidth variation',
        'Flatness mean','Flatness variation']
CSS="""
@font-face {font-family:'IBM Plex Mono';src:url('__FONT_BASE__/IBMPlexMono-Regular.ttf') format('truetype');font-weight:400;font-style:normal;font-display:swap;}
@font-face {font-family:'IBM Plex Mono';src:url('__FONT_BASE__/IBMPlexMono-SemiBold.ttf') format('truetype');font-weight:600;font-style:normal;font-display:swap;}
@font-face {font-family:'IBM Plex Mono';src:url('__FONT_BASE__/IBMPlexMono-Bold.ttf') format('truetype');font-weight:700;font-style:normal;font-display:swap;}
html,body {background:#0d0d0d!important;}
.gradio-container {max-width:1200px!important;margin:auto!important;padding:12px!important;font-family:'IBM Plex Mono',monospace!important;}
.gradio-container,.gradio-container * {font-family:'IBM Plex Mono',monospace!important;border-radius:0!important;box-shadow:none!important;background-image:none!important;backdrop-filter:none!important;text-shadow:none!important;}
.gradio-container *::before,.gradio-container *::after {border-radius:0!important;box-shadow:none!important;background-image:none!important;backdrop-filter:none!important;}
.gradio-container .hero {padding:16px 0 12px;border-bottom:1px solid #2a2a28;margin-bottom:8px;}
.gradio-container .eyebrow {font-size:11px;letter-spacing:2px;font-weight:700;color:#8a8a85;text-transform:uppercase;}
.gradio-container .hero h1 {font-size:42px;line-height:1.1;letter-spacing:-1.5px;color:#e8e8e3;margin:6px 0;}
.gradio-container .hero p {max-width:720px;color:#8a8a85;font-size:16px;line-height:1.6;margin:6px 0;}
.gradio-container .result-card {padding:12px;background:#0d0d0d;border:1px solid #2a2a28;}
.gradio-container .result-card h2 {font-size:26px;margin:4px 0 8px;color:#e8e8e3;}
.gradio-container .result-card p {color:#8a8a85;line-height:1.55;margin:6px 0;}
.gradio-container .result-card[data-result="ai"] {border-color:#d4a24e;--result-accent:#d4a24e;}
.gradio-container .result-card[data-result="human"] {border-color:#5a8a80;--result-accent:#5a8a80;}
.gradio-container .score-value {display:inline-block;min-width:7ch;text-align:right;font-variant-numeric:tabular-nums;color:#e8e8e3;}
.gradio-container .score-track {height:8px;background:#0d0d0d;border:1px solid #2a2a28;position:relative;margin:12px 0 6px;}
.gradio-container .score-pin {position:absolute;top:-4px;height:14px;width:4px;background:var(--result-accent,#e8e8e3);}
.gradio-container .score-labels {display:flex;justify-content:space-between;color:#8a8a85;font-size:12px;}
.gradio-container .stat-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0;}
.gradio-container .stat {border:1px solid #2a2a28;background:#0d0d0d;padding:10px;}
.gradio-container .stat strong {font-size:28px;display:block;color:#e8e8e3;text-align:right;font-variant-numeric:tabular-nums;}
.gradio-container .stat span {color:#8a8a85;font-size:13px;}
.gradio-container .small-note {font-size:13px;color:#8a8a85;line-height:1.6;}
.gradio-container [role="tablist"] {border-bottom:1px solid #2a2a28;gap:4px;}
.gradio-container [role="tab"] {padding:8px 10px;border:1px solid transparent;color:#8a8a85;}
.gradio-container [role="tab"][aria-selected="true"] {color:#e8e8e3;border-bottom:1px solid #5a8a80;}
.gradio-container button:focus-visible,.gradio-container input:focus-visible,.gradio-container textarea:focus-visible {outline:1px solid #5a8a80;outline-offset:1px;}
.gradio-container table {border-collapse:collapse;border:1px solid #2a2a28;}
.gradio-container th,.gradio-container td {border:1px solid #2a2a28;padding:6px 8px;}
.gradio-container #feature-values th:not(:first-child),.gradio-container #feature-values td:not(:first-child),.gradio-container #holdout-errors th:last-child,.gradio-container #holdout-errors td:last-child {text-align:right;font-variant-numeric:tabular-nums;}
.gradio-container hr {border:0;border-top:1px solid #2a2a28;}
@media(max-width:650px){.gradio-container .hero h1{font-size:30px}.gradio-container .stat-grid{grid-template-columns:1fr}.gradio-container .stat strong{font-size:24px}}
""".replace('__FONT_BASE__','/gradio_api/file='+FONT_DIR.as_posix())


def build_theme():
    """The same solid, monochrome chrome in light and dark browser preferences."""
    def solid(color,name):
        return gr.themes.colors.Color(*([color]*11),name=name)
    neutral=gr.themes.colors.Color(INK,INK,MUTED,MUTED,BORDER,BORDER,BORDER,BACKGROUND,BACKGROUND,BACKGROUND,BACKGROUND,name='technical-neutral')
    theme=gr.themes.Soft(primary_hue=solid(GREEN,'dull-teal'),secondary_hue=solid(CORAL,'amber'),
        neutral_hue=neutral,spacing_size='sm',font=['IBM Plex Mono','monospace'],font_mono=['IBM Plex Mono','monospace'])
    values={}
    supported=inspect.signature(theme.set).parameters
    def set_values(names,value):
        for name in names.split():
            values[name]=value
            if name+'_dark' in supported:values[name+'_dark']=value
    set_values('body_background_fill background_fill_primary background_fill_secondary block_background_fill panel_background_fill input_background_fill input_background_fill_hover input_background_fill_focus block_label_background_fill block_title_background_fill table_even_background_fill table_odd_background_fill',BACKGROUND)
    set_values('body_text_color block_label_text_color block_title_text_color accordion_text_color table_text_color',INK)
    set_values('body_text_color_subdued block_info_text_color input_placeholder_color',MUTED)
    set_values('border_color_primary block_border_color panel_border_color input_border_color table_border_color',BORDER)
    set_values('border_color_accent input_border_color_focus input_border_color_hover',GREEN)
    set_values('block_border_width panel_border_width input_border_width button_border_width','1px')
    set_values('block_radius container_radius input_radius table_radius embed_radius block_label_radius block_label_right_radius block_title_radius button_small_radius button_medium_radius button_large_radius','0px')
    set_values('shadow_drop shadow_drop_lg shadow_inset block_shadow block_label_shadow input_shadow input_shadow_focus','none')
    set_values('block_padding input_padding layout_gap form_gap_width','8px')
    set_values('button_large_padding button_medium_padding button_small_padding','8px 12px')
    for kind in ('primary','secondary'):
        prefix='button_'+kind+'_'
        for suffix,value in {'background_fill':BACKGROUND,'background_fill_hover':BACKGROUND,
            'text_color':INK,'text_color_hover':INK,'border_color':GREEN if kind=='primary' else BORDER,
            'border_color_hover':GREEN,'shadow':'none','shadow_hover':'none','shadow_active':'none'}.items():
            set_values(prefix+suffix,value)
    return theme.set(**values)


EMPTY='<div class="result-card"><div class="eyebrow">Ready when you are</div><h2>What does the model hear?</h2><p>Choose an audio file or a local example, then select Analyze recording.</p><p class="small-note">First 10 seconds · ten audio features · one trained classifier</p></div>'


def style_axis(axis):
    axis.spines[['top','right']].set_visible(False)
    axis.tick_params(colors=MUTED,labelsize=9)
    for spine in axis.spines.values():spine.set_color(BORDER);spine.set_linewidth(1)
    axis.set_axisbelow(True)
    axis.grid(color=BORDER,alpha=1,linewidth=1)


def signal_figure(clip):
    figure=Figure(figsize=(9,5.1),layout='constrained',facecolor=BACKGROUND)
    waveform,spectrum=figure.subplots(2,1)
    waveform.plot(np.arange(clip.size)/TARGET_SAMPLE_RATE,clip,color=GREEN,linewidth=.5)
    waveform.set(title='Waveform · the exact clip used by the model',xlabel='Time (seconds)',ylabel='Amplitude',xlim=(0,10))
    power,freq,times=spectrogram(Audio(clip[:,None],TARGET_SAMPLE_RATE))
    db=10*np.log10(np.maximum(power,1e-12))
    bounds=np.linspace(float(db.min()),float(db.max()) if db.max()>db.min() else float(db.min())+1,5)
    im=spectrum.pcolormesh(times,freq/1000,db,shading='auto',cmap=STEPPED,norm=BoundaryNorm(bounds,STEPPED.N))
    spectrum.set(title='Spectrogram · how frequency energy changes',xlabel='Time (seconds)',ylabel='Frequency (kHz)',xlim=(0,10))
    figure.colorbar(im,ax=spectrum,label='Power density (dB re 1 amplitude²/Hz)')
    for axis in (waveform,spectrum):style_axis(axis)
    return figure


def contribution_figure(contributions,intercept):
    values=np.append(contributions,intercept)
    labels=PRETTY+['Model intercept']
    order=np.argsort(values)
    fig=Figure(figsize=(9,4.8),layout='constrained',facecolor=BACKGROUND);axis=fig.subplots()
    axis.barh(np.asarray(labels)[order],values[order],color=[CORAL if v>=0 else GREEN for v in values[order]],height=.30)
    axis.axvline(0,color=INK,linewidth=.8)
    axis.set(xlabel='Contribution to the model logit · human direction ← 0 → AI direction',title='What moved this prediction?')
    style_axis(axis)
    return fig


def reset_results():
    return EMPTY,'',None,None,None,[]


def analyze(path):
    """Return real model measurements; errors clear previous results instead of leaving stale plots."""
    if not path:
        return '<div class="result-card"><h2>Add a recording first</h2><p>Choose a WAV, MP3, FLAC or OGG file of at least 10 seconds.</p></div>','',None,None,None,[]
    try:
        info=sf.info(path)
        if info.duration<10:
            raise ValueError(f'This recording is {info.duration:.2f} seconds long. Please choose one with at least 10 seconds.')
        if info.duration>300:
            raise ValueError('For this local demo, please choose a recording no longer than 5 minutes.')
        if info.frames*info.channels>60_000_000:
            raise ValueError('This recording is too large to decode in the demo. Please upload a shorter excerpt.')
        model=json.loads(MODEL_PATH.read_text(encoding='utf-8'))
        if any(digest(ROOT/name)!=model['contract_sha256'][name] for name in CONTRACT_FILES):
            raise ValueError('The audio pipeline has changed since training. Rebuild the baseline before using this model.')
        audio=load_audio(path)
        clip=preprocess_audio(audio,start_seconds=model['start_seconds'])
        vector=extract_features(clip)
        standardized=(vector-np.asarray(model['scaler_mean']))/np.asarray(model['scaler_scale'])
        contributions=standardized*np.asarray(model['coefficients'])
        score=float(score_features(model,vector))
        prediction='Likely AI-generated' if score>=model['threshold'] else 'Likely human-made'
        result=f'''<div class="result-card" data-result="{'ai' if score>=model['threshold'] else 'human'}"><div class="eyebrow">Baseline prediction</div><h2>{prediction}</h2>
        <p>AI score <strong class="score-value">{score:.3f}</strong> · decision threshold {model['threshold']:.2f}</p>
        <div class="score-track"><span class="score-pin" style="left:calc({score*100:.3f}% - 2px)"></span></div>
        <div class="score-labels"><span>0 · human direction</span><span>0.5</span><span>AI direction · 1</span></div>
        <p class="small-note">This is a model score, not a confidence percentage or proof of authorship. This baseline has no inconclusive outcome yet.</p></div>'''
        details=f'**{html.escape(Path(path).name)}** · {audio.duration_seconds:.2f} s · {audio.sample_rate:,} Hz · {audio.channels} channel(s)\n\nAnalyzed: first 10 seconds → 24 kHz mono → 10 features.'
        table=[[PRETTY[i],float(vector[i]),float(standardized[i]),float(contributions[i])] for i in range(10)]
        return result,details,str(path),signal_figure(clip),contribution_figure(contributions,model['intercept']),table
    except (ValueError,OSError,RuntimeError) as error:
        return f'<div class="result-card"><h2>Could not analyze this recording</h2><p>{html.escape(str(error))}</p></div>','',None,None,None,[]


def confusion_figure(metrics):
    figure=Figure(figsize=(10,3.5),layout='constrained',facecolor=BACKGROUND)
    axes=figure.subplots(1,3)
    for axis,role in zip(axes,['train','validation','holdout']):
        data=np.asarray(metrics[role]['confusion_matrix_true_rows_predicted_columns_human_ai'])
        axis.imshow(data,cmap=STEPPED,vmin=0,vmax=max(1,data.max()),interpolation='nearest')
        axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Human','AI'],yticklabels=['Human','AI'],
                 xlabel='Predicted label',ylabel='Dataset label',title=f'{role.title()} · {metrics[role]["tracks"]} tracks')
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
        ax.set(xlim=(-.03,1.03),ylim=(-.45,1.45),yticks=[0,1],yticklabels=['Human','AI'],xlabel='AI score',title=role.title())
        style_axis(ax)
    return fig


def local_examples():
    manifest=ROOT/'data/batch_manifest.csv'
    if not manifest.exists():return [],[]
    mapping={r['candidate_id']:r for r in read_csv(manifest)}
    predictions=read_csv(DEFAULT_OUTPUT/'predictions.csv')
    examples=[];labels=[]
    for true,pred,label in [('human','human','Human reference · correctly classified'),('ai','ai','Generated example · correctly classified'),('human','ai','Human reference · a known false positive')]:
        for r in predictions:
            source=mapping.get(r['candidate_id'])
            if r['true_label']==true and r['predicted_label']==pred and source and (ROOT/source['file_path']).exists():
                examples.append([str(ROOT/source['file_path'])]);labels.append(label+' — '+source['reference']);break
    return examples,labels


def build_demo():
    metrics=json.loads((DEFAULT_OUTPUT/'metrics.json').read_text(encoding='utf-8'))['metrics']
    predictions=read_csv(DEFAULT_OUTPUT/'predictions.csv')
    h=metrics['holdout'];cm=h['confusion_matrix_true_rows_predicted_columns_human_ai']
    with gr.Blocks(title='AI Music Detector · Audio Lab',analytics_enabled=False,delete_cache=(3600,3600)) as demo:
        gr.HTML('<div class="hero"><div class="eyebrow">Audio lab / Portfolio experiment 01</div><h1>Listen. Measure. Inspect.</h1><p>Explore how our trained AI music detector reads a recording—and see the measurements behind its decision.</p><span class="small-note">Handcrafted features + logistic regression · exploratory baseline</span></div>')
        with gr.Tab('Analyze audio'):
            with gr.Row():
                with gr.Column(scale=1,min_width=290):
                    upload=gr.File(label='Your recording',file_types=['.wav','.mp3','.flac','.ogg'],type='filepath',height=140)
                    gr.Markdown('**10 seconds to 5 minutes.** Only the first 10 seconds are analyzed. Uploads are processed on this computer; microphone evaluation is deferred.')
                    button=gr.Button('Analyze recording',variant='primary',size='lg')
                    original=gr.Audio(label='Listen to the original',interactive=False,format=None)
                    details=gr.Markdown('')
                with gr.Column(scale=2,min_width=360):
                    verdict=gr.HTML(EMPTY)
                    signal=gr.Plot(label='Inside the audio')
            with gr.Accordion('Why did the model lean this way?',open=True):
                gr.Markdown('Green bars push toward human; coral bars push toward AI. These are exact weighted contributions in **logit units**, including the intercept. They show model influence—not causal evidence of AI generation.')
                influence=gr.Plot(label='Feature contributions')
                with gr.Accordion('See the ten feature values',open=False):
                    table=gr.Dataframe(headers=['Feature','Measured value','Training-standardized value','Logit contribution'],datatype=['str','number','number','number'],interactive=False,elem_id='feature-values')
            outputs=[verdict,details,original,signal,influence,table]
            # Clear old results whenever the selected input changes.
            upload.change(reset_results,outputs=outputs,queue=False,api_name=False)
            button.click(analyze,inputs=upload,outputs=outputs,api_name='analyze',concurrency_limit=1)
            examples,labels=local_examples()
            if examples:
                gr.Examples(examples,inputs=[upload],label='Try an existing recording, then press Analyze',example_labels=labels,cache_examples=False)
        with gr.Tab('Model results'):
            gr.Markdown('## A real baseline, including its mistakes\nFour artist groups form this **exploratory holdout**: four human references and 12 generated recordings. The batch was inspected during development; it is not an untouched final test.')
            gr.HTML(f'<div class="stat-grid"><div class="stat"><strong>{cm[1][1]} / {h["ai_tracks"]}</strong><span>AI recordings detected · recall {h["ai_recall"]:.1%}</span></div><div class="stat"><strong>{cm[0][1]} / {h["human_tracks"]}</strong><span>Human recordings falsely flagged · {h["human_false_positive_rate"]:.0%}</span></div><div class="stat"><strong>{h["ai_precision"]:.1%}</strong><span>AI precision on this small holdout</span></div></div>')
            gr.Plot(confusion_figure(metrics),label='Correct predictions and errors')
            gr.Markdown('### Where the scores fall\nEach dot is one recording. The dashed line is the fixed 0.5 threshold. A human dot on the right is a false positive; an AI dot on the left is a missed detection.')
            gr.Plot(distribution_figure(predictions),label='Score distributions by dataset label')
            errors=[[r['reference'],r['generator'] or 'Human reference',r['true_label'],r['predicted_label'],float(r['ai_score'])] for r in predictions if r['split']=='holdout' and r['true_label']!=r['predicted_label']]
            gr.Dataframe(errors,headers=['Holdout recording','Source label','Dataset label','Prediction','AI score'],interactive=False,label='The actual holdout mistakes',elem_id='holdout-errors')
            gr.Markdown('**What this does not establish:** unseen-generator performance, microphone robustness or certified authorship. Source bandwidth, encoding, offsets and the small dataset may affect predictions. Human false positives are the main measured weakness.')
        with gr.Tab('How it works'):
            gr.Markdown('''## From sound to a score
**1 · Standardize the input.** Decode the recording, average its channels, resample to 24 kHz, and take the first ten seconds: 240,000 samples.

**2 · Measure the sound.** RMS amplitude, zero crossings, spectral center, bandwidth and flatness. A mean and standard deviation for each give ten features.

**3 · Apply what was learned.** The saved scaler uses training means and scales. Logistic regression combines ten weighted features with an intercept; a sigmoid maps that sum to an AI score.

**4 · Make the baseline decision.** Scores at or above 0.5 produce “Likely AI-generated.” Lower scores produce “Likely human-made.” The score is not calibrated and there is no inconclusive rule yet.

### The experiment behind this demo
80 recordings across 20 provisional artist groups: **48 training, 16 validation, 16 holdout**. Related reference/artist recordings remain together. Only training data fits the scaler and classifier. Historical FMA recordings supply human-reference labels; Echoes TTA supplies generated labels from ACE-Step, AudioLDM and MusicGen.

### Reading the visuals
The waveform and spectrogram show the exact processed clip used for prediction. The classifier receives the ten numeric summaries, rather than a spectrogram image. The feature chart shows each standardized feature multiplied by its learned weight; its bars plus the intercept equal the logit before the sigmoid.

**Built with** NumPy, SciPy, librosa, scikit-learn and Gradio. Dataset provenance, settings and limitations are recorded in the repository. No Hugging Face encoder is used in this version.
''')
    return demo


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    demo=build_demo()
    demo.queue(max_size=8).launch(server_name='127.0.0.1',server_port=args.port,share=False,
        inbrowser=False,theme=build_theme(),
        css=CSS,max_file_size='50mb',footer_links=[],allowed_paths=[str(ROOT/'.gradio_cache')])


if __name__=='__main__':main()
