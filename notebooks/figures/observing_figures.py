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
# This notebook makes the figures showing analysis of the observing dataset:
# - Figure 5
# - Figure 6
# - Supplementary Figure 6

# %% [markdown]
# # Load and set things

# %%
import os

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'observing.db'))

# %%
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

import matplotlib as mpl
import matplotlib.transforms as mtransforms

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
dest_fig_dir = figures_root_dir

# %%
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

# %% [markdown] tags=[]
# # Load factors and states

# %%
gpfa_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), 'gpfa_observing')

# %%
from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states, add_playing_observing_states

K = 18
transitions = 'sticky'

_, bin_length = load_results(K, transitions, gpfa_dir)

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)

# %%
add_playing_observing_states()

# %% [markdown]
# # Relation of observing role and HMM states 

# %% [markdown]
# ## P(role | state) 

# %%
from hidenseek.figure_util.observing.conditional_probs import *

# %%
probs = [(session.id, state_id, p_role_given_state_in_session(session, state_id, 'observing'), p_role_given_state_in_session(session, state_id, 'playing'))
         for session in Session.select()
         for state_id in range(K)]

# %%
probs_df = pd.DataFrame(probs, columns = ['session_id', 'state_id', 'observing', 'playing'])

# %% [markdown]
# ## Calculate real and fake scores 

# %%
from hidenseek.figure_util.fake_state_functions import *

# %%
import sklearn.metrics

score_fun = sklearn.metrics.mutual_info_score
score_kwargs = {}

N = 10000
seed = 123

np.random.seed(seed)


# %% tags=[]
generate_fake_states(generate_fake_states_by_rearranging_states, N, seed)

# %%
mi_tuples = []

# calculate real and fake scores session-wise
for session in tqdm(Session.select()):
    for used_trials in ['both', 'playing', 'observing']:
        for behavior_type in ['both', 'playing', 'observing']:
            if used_trials in ['playing', 'both'] and behavior_type == 'observing':
                # playing trials don't have observing behaviors
                continue

            # include trial's states if we're looking at all trials or the trial's type
            hmm_states = np.concatenate([trial.states for trial in session.trials if used_trials in ('both', trial.observing_role)])
            
            if behavior_type == 'both':
                # the behavior state is playing_states in playing trials and observing_states in observing trials
                behavior_states = np.concatenate([getattr(trial, f"{trial.observing_role}_states") for trial in session.trials if used_trials in ('both', trial.observing_role)])
            else:
                behavior_states = np.concatenate([getattr(trial, f"{behavior_type}_states") for trial in session.trials if used_trials in ('both', trial.observing_role)])

            real_score = score_fun(behavior_states, hmm_states, **score_kwargs)
            fake_states_list = [np.concatenate([trial.fake_states_list[i] for trial in session.trials if used_trials in ('both', trial.observing_role)])
                                for i in range(N)]

            mi_tuples.append((session.id, used_trials, behavior_type, 'real', real_score))
            for fake_states in fake_states_list:
                sc = score_fun(behavior_states, fake_states, **score_kwargs)
                mi_tuples.append((session.id, used_trials, behavior_type, 'fake', sc))
                
df = pd.DataFrame(mi_tuples, columns = ('session_id', 'used_trials', 'behavior_type', 'kind', 'score'))

# %% [markdown]
# # UMAP 

# %%
from hidenseek.figure_util.umap_embedding import embed_with_umap

# %%
session = Session[12]

embed_with_umap(session, 3, 40)

# %%
from hidenseek.util.plotting import label_subfigures

from hidenseek.figure_util.observing.__plot_most_likely_states import make_fig_most_likely_states
from hidenseek.figure_util.observing.__plot_probs import plot_probs_swarm

# %%
narrow_format = False

# %%
if narrow_format:
    fig = plt.figure(figsize = (narrow_cb_width, 0.6 * cb_height), constrained_layout = False, tight_layout = False)
else:
    fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = False, tight_layout = False)

gs = fig.add_gridspec(nrows = 3, ncols = 1, height_ratios = [3, 3, 4], hspace = 0.4)

# most likely states in one session (#12)
gs_single_session = gs[0].subgridspec(nrows = 1, ncols = 2)
gs_seek = gs_single_session[0].subgridspec(nrows = 2, ncols = 1)
gs_hide = gs_single_session[1].subgridspec(nrows = 2, ncols = 1)

ax_seek_play = fig.add_subplot(gs_seek[0])
ax_seek_obs = fig.add_subplot(gs_seek[1])
ax_hide_play = fig.add_subplot(gs_hide[0])
ax_hide_obs = fig.add_subplot(gs_hide[1])
big_ax_seek = fig.add_subplot(gs_seek[:], frame_on = False, xticks = [], yticks = [])
big_ax_hide = fig.add_subplot(gs_hide[:], frame_on = False, xticks = [], yticks = [])
big_ax_single_session = fig.add_subplot(gs_single_session[:], frame_on = False, xticks = [], yticks = [])

# separation of the states based on P(obs | state)
# and mutual info of state labels and role
ax_obs_prob_swarm = fig.add_subplot(gs[1])

# for the labels
big_ax_obs_prob_swarm = fig.add_subplot(gs[1], frame_on = False, xticks = [], yticks = [])

# make plots
ticklabels = make_fig_most_likely_states(session, ax_seek_obs, ax_seek_play, ax_hide_obs, ax_hide_play, big_ax_seek, big_ax_hide, K, bin_length)
ax_seek_obs.set_xlabel('warped time (s)')
ax_hide_obs.set_xlabel('warped time (s)')
plot_probs_swarm(probs_df, ax_obs_prob_swarm, size=2)

ax_obs_prob_swarm.set_xlabel("session ID (observing)")


gs_umap = gs[2].subgridspec(nrows = 1, ncols = 2)
ax_playing = fig.add_subplot(gs_umap[0], projection='3d')
ax_observing = fig.add_subplot(gs_umap[1], projection='3d', sharex = ax_playing, sharey = ax_playing, sharez = ax_playing)
ax_umap_label = fig.add_subplot(gs_umap[:], frame_on = False, xticks = [], yticks = [])

for trial in session.playing_trials:
    ax_playing.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = [state_colors[k] for k in trial.states.values], s = 0.5)
    
for trial in session.observing_trials:
    ax_observing.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = [state_colors[k] for k in trial.states.values], s = 0.5)

ax_playing.set_title('Playing', pad = 0)
ax_observing.set_title('Observing', pad = 0)

ax_playing.set(xticklabels = [], yticklabels = [], zticklabels = [])
ax_observing.set(xticklabels = [], yticklabels = [], zticklabels = [])

labelpad = -10
for axi in (ax_playing, ax_observing):
    axi.set_xlabel('$x_1$', labelpad = labelpad)
    axi.set_ylabel('$x_2$', labelpad = labelpad)
    axi.set_zlabel('$x_3$', labelpad = labelpad)


if narrow_format:
    x_offs = -0.125
else:
    x_offs = -0.1
    
annotations = label_subfigures([big_ax_single_session, big_ax_obs_prob_swarm, ax_umap_label], 'auto', x_offs)

fig.align_ylabels([big_ax_single_session, big_ax_obs_prob_swarm, ax_umap_label])
fig.canvas.draw()

## move the plots in A so that the top of the time point names aligns with the top of the A label

# get the height of the ABCD label
ann_box = annotations[0].get_window_extent().inverse_transformed(fig.transFigure)
ann_height = ann_box.y1 - ann_box.y0

# get the height of the time point labels
bboxes = []
for label in ticklabels:
    bbox = label.get_window_extent()
    # the figure transform goes from relative coords->pixels and we want the inverse of that
    bboxi = bbox.transformed(fig.transFigure.inverted())
    bboxes.append(bboxi)

# this is the bbox that bounds all the bboxes, again in relative figure coords
bbox = mtransforms.Bbox.union(bboxes)

# move the state probability axes so that the top of the time point labels
# lines up with the top of the ABCD label
moving_height = bbox.height - ann_height

# actually do the moving
for axi in [ax_seek_obs, ax_seek_play, ax_hide_obs, ax_hide_play, big_ax_seek, big_ax_hide, ax_obs_prob_swarm, big_ax_obs_prob_swarm, ax_playing, ax_observing, ax_umap_label]:
    ax_box = axi.get_position()
    width = ax_box.x1 - ax_box.x0
    height = ax_box.y1 - ax_box.y0
    axi.set_position([ax_box.x0, ax_box.y0 - moving_height, width, height])

# %%
if narrow_format:
    fig.savefig(os.path.join(dest_fig_dir, f'observing_fig_K_{K}_narrow.png'), dpi = 400)
    fig.savefig(os.path.join(dest_fig_dir, f'observing_fig_K_{K}_narrow.pdf'), dpi = 400)
else:
    fig.savefig(os.path.join(dest_fig_dir, f'observing_fig_K_{K}.png'), dpi = 400)
    fig.savefig(os.path.join(dest_fig_dir, f'observing_fig_K_{K}.pdf'), dpi = 400)

# %% [markdown]
# # MI plots 

# %%
N = 10000

scores = []
for session in tqdm(Session.select()):
    role_states = np.concatenate([np.ones_like(trial.states) if trial.observing else np.zeros_like(trial.states) for trial in session.trials])
    real_states = np.concatenate([trial.states for trial in session.trials])
    
    fake_states_list = generate_fake_states_by_shuffling_labels(real_states, N, seed)

    scores.append((session.id, sklearn.metrics.mutual_info_score(role_states, real_states), 'real'))
    for states in fake_states_list:
        scores.append((session.id, sklearn.metrics.mutual_info_score(role_states, states), 'fake'))
    
role_mi_df = pd.DataFrame(scores, columns = ['session_id', 'score', 'kind'])

# %%
# playing rat's behavior on playing trials
playing_df = df.query('used_trials == "playing" and behavior_type == "playing"')

# recorded rat's behavior on all trials
both_df = df.query('used_trials == "both" and behavior_type == "both"')

# observing rat's behavior on observing trials
observing_df = df.query('used_trials == "observing" and behavior_type == "observing"')

# playing behaviors of the other rat on observing trials
mixed_df = df.query('used_trials == "observing" and behavior_type == "playing"')

# %%
from hidenseek.figure_util.contingency import *

p_value = 5 / 100
correction = 'holm'

cont_df, cont_df_hmm, cont_df_beh, signif_cont_df, signif_cont_df_hmm, signif_cont_df_beh = make_cont_dfs(session,
                                                                                                          'playing',
                                                                                                          'playing',
                                                                                                          p_value,
                                                                                                          rename=True,
                                                                                                          correction = correction)

# %%
from hidenseek.figure_util.observing.__plot_mi import plot_mi
from hidenseek.figure_util.contingency_plotting import make_fig_hmm_behavior_conditional_probs

fig = plt.figure(figsize = (cb_width, 0.7 * cb_height), constrained_layout = False)

gs = fig.add_gridspec(nrows = 1, ncols = 2, width_ratios = [3, 2], wspace = 0.3)

gs_left = gs[0].subgridspec(nrows = 3, ncols = 1, hspace = 0.5)

ax_playing = fig.add_subplot(gs_left[0])
ax_observing = fig.add_subplot(gs_left[1])
ax_mixed = fig.add_subplot(gs_left[2])

scale_orange_dot = 0.5
plot_mi(playing_df, ax_playing, scale = scale_orange_dot)
plot_mi(observing_df, ax_observing, scale = scale_orange_dot)
plot_mi(mixed_df, ax_mixed, scale = scale_orange_dot)

ax_playing.set_title('play behaviors on playing trials')
ax_observing.set_title("observing behaviors on observing trials")
ax_mixed.set_title("other (playing) rat's behavior on observing trials")

for axi in [ax_playing, ax_observing, ax_mixed]:
    axi.set_ylabel('MI')
    sns.despine(ax = axi)
    axi.set_xlabel("session ID (observing)")
    
gs_right = gs[1].subgridspec(nrows = 3, ncols = 1, height_ratios = [20, 20, 1], hspace = 1.2)

ax_cont_hmm = fig.add_subplot(gs_right[0])
ax_cont_beh = fig.add_subplot(gs_right[1])
ax_cont_cbar = fig.add_subplot(gs_right[2])

ax_playing_label = fig.add_subplot(gs_left[0], xticks = [], yticks = [], frame_on = False, label = 'playing_label')
ax_observing_label = fig.add_subplot(gs_left[1], xticks = [], yticks = [], frame_on = False, label = 'observing_label')
ax_mixed_label = fig.add_subplot(gs_left[2], xticks = [], yticks = [], frame_on = False, label = 'mixed_label')
ax_cont_label = fig.add_subplot(gs_right[:], xticks = [], yticks = [], frame_on = False)

make_fig_hmm_behavior_conditional_probs(cont_df_hmm.T,
                                        cont_df_beh.T,
                                        ax_cont_hmm,
                                        ax_cont_beh,
                                        ax_cont_cbar,
                                        state_colors,
                                        signif_cont_df_hmm.T,
                                        signif_cont_df_beh.T,
                                        cbar_orient = 'horizontal',
                                        remove_top_xticks = False)
ax_cont_cbar.set_xlabel('Conditional probability')

# add ABDC labels and align them
x_offs = -0.1
annotations = label_subfigures([ax_playing_label, ax_observing_label, ax_mixed_label, ax_cont_label], [x_offs, x_offs, x_offs, 1.5*x_offs])

fig.align_ylabels([ax_playing_label, ax_observing_label, ax_mixed_label])

# %%
fig.savefig(os.path.join(dest_fig_dir, f'fig_observing_MI_K_{K}_p_{p_value:.5f}_N_{N}_{correction}_correction_without_role_mi.png'), dpi = 400)
fig.savefig(os.path.join(dest_fig_dir, f'fig_observing_MI_K_{K}_p_{p_value:.5f}_N_{N}_{correction}_correction_without_role_mi.pdf'), dpi = 400)

# %% [markdown]
# # MI with the observing vs playing role 

# %%
fig, ax = plt.subplots(figsize = (cb_width, cb_height / 3))

plot_mi(role_mi_df, ax, 0.8)
ax.set_xlabel("session ID (observing)")
sns.despine(ax = ax)

# %%
fig.savefig(os.path.join(dest_fig_dir, f'supp_fig_observing_role_MI.png'), dpi = 400)
fig.savefig(os.path.join(dest_fig_dir, f'supp_fig_observing_role_MI.pdf'), dpi = 400)

# %% [markdown]
# Some analysis about p-values

# %%
for (sid, sdf) in role_mi_df.groupby('session_id'):
    real_score = sdf.query('kind == "real"').score.item()
    fake_scores = sdf.query('kind == "fake"').score.values
    
    print(np.mean(fake_scores > real_score))

# %%
from hidenseek.figure_util.get_p import get_p

print('playing:', [get_p(sdf) for (_, sdf) in playing_df.groupby('session_id')], sep = '\t')
print('both:', [get_p(sdf) for (_, sdf) in both_df.groupby('session_id')], sep = '\t\t')
print('observing:', [get_p(sdf) for (_, sdf) in observing_df.groupby('session_id')], sep = '\t')
print('mixed:', [get_p(sdf) for (_, sdf) in mixed_df.groupby('session_id')], sep = '\t\t')

# %%
p_threshold = 5 / N

for (dfi, label) in zip([playing_df, both_df, observing_df, mixed_df],
                        ['playing', 'both', 'observing', 'mixed']):
    print(f'{label:<10}', sum(np.array([get_p(sdf) for (_, sdf) in dfi.groupby('session_id')]) < p_threshold), sep = '\t')

# %%
print([get_p(sdf) for (_, sdf) in role_mi_df.groupby('session_id')])

# %%
[get_p(sdf) for (_, sdf) in playing_df.groupby('session_id')]

# %%
