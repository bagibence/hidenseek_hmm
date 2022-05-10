# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python (hidenseek)
#     language: python
#     name: hidenseek
# ---

# %% [markdown]
# This notebook extracts the parts of the videos for each state where that state was active.
#
# Video files are not included.

# %% [markdown]
# # Imports and setup

# %%
import pickle
import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as scs
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns
# %matplotlib inline

from tqdm.autonotebook import tqdm

# %%
from moviepy.editor import *
import proglog
proglog.notebook()
from moviepy.video.io.bindings import mplfig_to_npimage

from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.video import *

from hidenseek.util.plotting import get_tab20_and_norm, get_state_colors, add_vertical_lines_for_time_points

from hidenseek.util.postproc.transitions import *
from hidenseek.util.postproc.misc import correct_starts_and_ends

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %% [markdown]
# # Load HMM data

# %%
gpfa_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), 'gpfa')

# %%
K = 11
transitions = 'sticky'
n_seeds = 40

load_results(K, transitions, gpfa_dir=gpfa_dir)

# %%
tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)

# %% [markdown]
# # Select session and states

# %%
session = Session[13]

states_to_do = range(K)

# %% [markdown]
# # Get set source and target paths

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
figures_dir = os.path.join(figures_root_dir, f'GPFA_{K}_states')

for d in [figures_root_dir, figures_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
# %%
source_video_dir = f'/Volumes/Maxtor/HideNSeek/{session.animal}/{session.date}/Bonsai'
video_filename1 = [fname for fname in  os.listdir(source_video_dir) if fname.startswith("VideoCam1") and fname.endswith('.avi')][0]
video_filename2 = [fname for fname in  os.listdir(source_video_dir) if fname.startswith("VideoCam2") and fname.endswith('.avi')][0]

# %%
clip1 = VideoFileClip(os.path.join(source_video_dir, video_filename1))
clip2 = VideoFileClip(os.path.join(source_video_dir, video_filename2))

# %%
base_video_dir = f'/Volumes/Maxtor/HideNSeek/extracted_clips_review'

animal_dir   = os.path.join(base_video_dir, session.animal)
session_dir  = os.path.join(animal_dir, session.date)
gpfa_out_dir = os.path.join(session_dir, 'GPFA')
out_dir      = os.path.join(gpfa_out_dir, f'HMM_{transitions}_{K}')

for dirpath in [base_video_dir, animal_dir, session_dir, gpfa_out_dir, out_dir]:
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

# %%
if not os.path.exists(os.path.join(out_dir, 'states')):
    os.mkdir(os.path.join(out_dir, 'states'))

# %%
print(f'Reading from {gpfa_dir}')
print(f'Saving things to {out_dir}')

# %% [markdown]
# # Define a helper function 

# %%
import io
from hidenseek.util.plotting.plotting_setup import *
#plt.rcParams['text.antialiased'] = True

def make_span_im(trial, start, end, state_id):
    """
    Create a figure with the time points of the trial
    and shade from start to end using the color associated
    with state_id
    """
    width = A4_width
    height = width / 4
    fig, ax = plt.subplots(figsize = (width, height), tight_layout=True)
    ax.set_xlim((0, trial.time_points.end))
    add_vertical_lines_for_time_points(ax, trial.time_points.values, 'black', linewidth = 2)
    ax.axvspan(start, end, color = state_colors[state_id])
    ax.set_yticklabels([])
    ax.set_xticks(trial.time_points.values)
    
    if trial.role == 'seek':
        trial_num = trial.session.seek_trials.index(trial)
    else:
        trial_num = trial.session.hide_trials.index(trial)
    ax.set_xlabel(f'Trial {trial_num + 1} ({trial.role})', size = 18)
    if trial.role == 'seek':
        ax.set_xticklabels(['st', 'BO', 'JO', 'int', 'tran', 'JI', 'end'], size = 12, rotation = 60)
    elif trial.role == 'hide':
        ax.set_xticklabels(['st', 'JO', 'int', 'tran', 'JI', 'end'], size = 12, rotation = 60)
        
    with io.BytesIO() as buff:
        fig.patch.set_facecolor('white')
        fig.savefig(buff, format='png')
        buff.seek(0)
        pic = plt.imread(buff)
    #pic = mplfig_to_npimage(fig)
    plt.close(fig)
    
    return (pic * 255).astype(int)
    #return pic


# %%
fig, ax = plt.subplots(figsize = (20, 4))
ax.imshow(make_span_im(session.seek_trials[1], 10000, 20000, 5))
fig.patch.set_facecolor('black')

# %% [markdown]
# ## For each state, see what happened when it was active 

# %%
pad_start = 0.
pad_end = 0.

# %% [markdown]
# ### One video per trial and state 

# %% tags=[] jupyter={"outputs_hidden": true}
for state_id in tqdm(states_to_do):
    state_dir = os.path.join(out_dir, 'states', f'state_{state_id}')
    save_dir = os.path.join(state_dir, 'per_trial')
    seek_dir = os.path.join(save_dir, 'seek')
    hide_dir = os.path.join(save_dir, 'hide')
    for dirname in [state_dir, save_dir, seek_dir, hide_dir]:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    # concatenating per trial
    for trial in tqdm(session.trials):
        pairs = get_transition_state_pairs(trial.states)
        transition_times = get_transition_points(trial.states)

        starts, ends = [], []
        for p, t in zip(pairs, transition_times):
            if p[1] == state_id:
                starts.append(t)
            if p[0] == state_id:
                ends.append(t)

        starts, ends = correct_starts_and_ends(starts, ends)

        # add pads
        for i in range(len(starts)):
            starts[i] = max(0, starts[i] - pad_start)
        for i in range(len(ends)):
            if ends[i] is None:
                ends[i] = trial.time_points.end
            else:
                ends[i] = min(trial.time_points.end, ends[i] + pad_end)

        abs_starts = np.array(starts) + trial.abs_time_points.start
        abs_ends = np.array(ends) + trial.abs_time_points.start

        starts1 = [neural_to_video_time(t, session.frametimes1, clip1.fps) for t in abs_starts]
        starts2 = [neural_to_video_time(t, session.frametimes2, clip2.fps) for t in abs_starts]
        ends1 = [neural_to_video_time(t, session.frametimes1, clip1.fps) for t in abs_ends]
        ends2 = [neural_to_video_time(t, session.frametimes2, clip2.fps) for t in abs_ends]

        assert all([s <= e for (s, e) in zip(starts1, ends1)])
        assert all([s <= e for (s, e) in zip(starts2, ends2)])

        durations = [e - s for s,e in zip(starts1, ends1)]
        spans = [ImageClip(make_span_im(trial, s, e, state_id), duration = d).resize(width = clip1.size[0])
                 for s, e, d in zip(starts, ends, durations)]

        if len(starts1) > 0:
            if trial.role == 'seek':
                out_filename = os.path.join(seek_dir, f'trial{trial.id}.mp4')
            elif trial.role == 'hide':
                out_filename = os.path.join(hide_dir, f'trial{trial.id}.mp4')
            concatenate_videoclips([clips_array([[clip2.subclip(s2, e2)],
                                                 [clip1.subclip(s1, e1)],
                                                 [span]])
                                    for (s1,s2,e1,e2,span) in zip(starts1, starts2, ends1, ends2,spans)]).write_videofile(out_filename, preset='ultrafast', codec='libx264', remove_temp=True, audio=False, fps=30)

# %% [markdown]
# ### One video per state and role
#
# For each state, create a video in seek and another in hide

# %%
# create one video per state

for state_id in tqdm(states_to_do):
    seek_vids = []
    hide_vids = []
    state_dir = os.path.join(out_dir, 'states', f'state_{state_id}')
    for dirname in [state_dir]:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    # concatenating per trial
    for trial in tqdm(session.trials):
        pairs = get_transition_state_pairs(trial.states)
        transition_times = get_transition_points(trial.states)

        starts, ends = [], []
        for p, t in zip(pairs, transition_times):
            if p[1] == state_id:
                starts.append(t)
            if p[0] == state_id:
                ends.append(t)

        starts, ends = correct_starts_and_ends(starts, ends)

        # add pads
        for i in range(len(starts)):
            starts[i] = max(0, starts[i] - pad_start)
        for i in range(len(ends)):
            if ends[i] is None:
                ends[i] = trial.time_points.end
            else:
                ends[i] = min(trial.time_points.end, ends[i] + pad_end)

        abs_starts = np.array(starts) + trial.abs_time_points.start
        abs_ends = np.array(ends) + trial.abs_time_points.start

        starts1 = [neural_to_video_time(t, session.frametimes1, clip1.fps) for t in abs_starts]
        starts2 = [neural_to_video_time(t, session.frametimes2, clip2.fps) for t in abs_starts]
        ends1 = [neural_to_video_time(t, session.frametimes1, clip1.fps) for t in abs_ends]
        ends2 = [neural_to_video_time(t, session.frametimes2, clip2.fps) for t in abs_ends]

        assert all([s < e for (s, e) in zip(starts1, ends1)])
        assert all([s < e for (s, e) in zip(starts2, ends2)])

        durations = [e - s for s,e in zip(starts1, ends1)]
        spans = [ImageClip(make_span_im(trial, s, e, state_id), duration = d).resize(width = clip1.size[0])
                 for s, e, d in zip(starts, ends, durations)]

        if len(starts1) > 0:
            trial_vid = concatenate_videoclips([clips_array([[clip2.subclip(s2, e2)],
                                                             [clip1.subclip(s1, e1)],
                                                             [span]])
                                                for (s1,s2,e1,e2,span) in zip(starts1, starts2, ends1, ends2,spans)])
            if trial.role == 'seek':
                seek_vids.append(trial_vid)
                seek_vids.append(ColorClip(trial_vid.size, (0,0,0), duration=0.5))
            elif trial.role == 'hide':
                hide_vids.append(trial_vid)
                hide_vids.append(ColorClip(trial_vid.size, (0,0,0), duration=0.5))
            
            
    concatenate_videoclips(seek_vids).write_videofile(os.path.join(state_dir, 'seek.mp4'), preset='medium', codec='libx264', remove_temp=True, audio=False, fps=30)
    concatenate_videoclips(hide_vids).write_videofile(os.path.join(state_dir, 'hide.mp4'), preset='medium', codec='libx264', remove_temp=True, audio=False, fps=30)

# %% [markdown]
# ## For every trial, show the whole trial and which state was active at every time point

# %%
state_colors = get_state_colors(K)

# %%
trials_dir = os.path.join(out_dir, 'trials')
seek_dir = os.path.join(trials_dir, 'seek')
hide_dir = os.path.join(trials_dir, 'hide')

for dirname in [trials_dir, seek_dir, hide_dir]:
    if not os.path.exists(dirname):
        os.mkdir(dirname)

# %%
for trial in tqdm(session.trials):
    trial_vid1 = clip1.subclip(neural_to_video_time(trial.abs_time_points.start, session.frametimes1, clip1.fps),
                               neural_to_video_time(trial.abs_time_points.end, session.frametimes1, clip1.fps))
    trial_vid2 = clip2.subclip(neural_to_video_time(trial.abs_time_points.start, session.frametimes2, clip2.fps),
                               neural_to_video_time(trial.abs_time_points.end, session.frametimes2, clip2.fps))

    pairs = get_transition_state_pairs(trial.states)
    pairs.append((pairs[-1][1], pairs[-1][1]))
    transition_times = get_transition_points(trial.states)
    transition_times = np.insert(transition_times, 0, 0)
    transition_times = np.append(transition_times, trial.time_points.end)

    start_times = transition_times[:-1]
    end_times = transition_times[1:]

    spans = [ImageClip(make_span_im(trial, s, e, int(p[0])), duration = (e - s) / 1000).resize(width = clip1.size[0])
             for s, e, p in zip(start_times, end_times, pairs)]

    state_indicator = concatenate_videoclips(spans)
    
    if trial.role == 'seek':
        output_dir = seek_dir
    elif trial.role == 'hide':
        output_dir = hide_dir
        
    filename = os.path.join(output_dir, f'trial_{trial.id}.mp4')
    clips_array([[trial_vid2],
                 [trial_vid1],
                 [state_indicator]]).write_videofile(filename, preset = 'ultrafast', codec='libx264', audio = False, remove_temp = True, fps = 30)

# %%
