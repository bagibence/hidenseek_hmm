# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# This notebook generates Supplementary Figure 1.

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

import pickle
from joblib import Parallel, delayed

import ssm
import autograd.numpy.random as npr

# %%
from hidenseek.figure_util.load_results import load_results, load_factors
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states
from hidenseek.util.plotting.plotting_setup import *

from hidenseek.util.plotting import get_tab20_and_norm, label_subfigures, get_state_colors

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %% [markdown]
# # Load states 

# %%
K = 11
transitions = 'sticky'

_, bin_length = load_results(K, 'sticky')

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)

# %%
add_behavioral_states()

# %% [markdown]
# # Embed with UMAP 

# %%
from hidenseek.figure_util.umap_embedding import embed_with_umap

# %%
session = Session[13]

embed_with_umap(session)

# %% [markdown]
# # Calculate conditional probabilities

# %%
p_tuples = []
for state in range(K):
    for role in ['seek', 'hide']:
        T_role = sum([trial.states.time.size for trial in session.trials if trial.role == role])
        T_state = sum([np.sum(trial.states.values == state) for trial in session.trials])
        T_role_and_state = sum([np.sum(trial.states.values == state) for trial in session.trials if trial.role == role])

        P_state_given_role = T_role_and_state / T_role
        P_role_given_state = T_role_and_state / T_state
        
        p_tuples.append((state, role, P_state_given_role, P_role_given_state))
        
pdf = pd.DataFrame(p_tuples, columns = ('state', 'role', 'P_state_given_role', 'P_role_given_state'))

# %%
T_seek = sum([trial.states.time.size for trial in session.trials if trial.role == 'seek'])
T_hide = sum([trial.states.time.size for trial in session.trials if trial.role == 'hide'])

P_seek = T_seek / (T_seek + T_role)
P_hide = T_hide / (T_seek + T_role)

def normalize_by_role_length(row):
    if row.role == 'seek':
        return row.P_role_given_state / P_seek
    else:
        return row.P_role_given_state / P_hide
    
pdf['P_role_given_state_normalized'] = pdf.apply(normalize_by_role_length, axis = 1)

# %%
seek_color = 'violet'
hide_color = 'lightgreen'


# %%
def make_fig_umap_3d_hide_and_seek(session, ax_seek, ax_hide):
    for trial in session.seek_trials:
        ax_seek.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = seek_color, s = 0.5)
    for trial in session.hide_trials:
        ax_hide.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = hide_color, s = 0.5)

    labelpad = -15
    for ax in [ax_seek, ax_hide]:
        ax.set_xlabel('$x_1$', labelpad = labelpad)
        ax.set_ylabel('$x_2$', labelpad = labelpad)
        ax.set_zlabel('$x_3$', labelpad = labelpad)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    
def make_fig_umap_3d_hide_and_seek_single_axis(session, ax, alpha = 0.2, s = None):
    ax.scatter(*np.row_stack([trial.embedding.values for trial in session.seek_trials]).T, color = seek_color, label = 'seek', alpha = alpha, s = s)
    ax.scatter(*np.row_stack([trial.embedding.values for trial in session.hide_trials]).T, color = hide_color, label = 'hide', alpha = alpha, s = s)

    labelpad = -15
    ax.set_xlabel('$x_1$', labelpad = labelpad)
    ax.set_ylabel('$x_2$', labelpad = labelpad)
    ax.set_zlabel('$x_3$', labelpad = labelpad)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.legend()


# %%
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting.colors import parula_map
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

from hidenseek.figure_util.MI_distribution import *
from hidenseek.figure_util.linear_states_in_session import *
from hidenseek.figure_util.single_trial_rates_factors_states import *

import matplotlib.transforms as mtransforms

from hidenseek.util.plotting.plotting_setup import *

# %% [markdown] tags=[]
# # State segment lengths 

# %%
from hidenseek.util.postproc import get_transition_indices

# %%
session = Session[13]

# %%
transition_indices = get_transition_indices(session.states)
start_indices = np.insert(transition_indices, 0, 0)

# might have to be size-1. probably not
state_lengths = np.diff(np.append(start_indices, session.states.size))

# states at the start of each segment
if isinstance(session.states, xr.DataArray):
    state_ids = session.states.values[start_indices]
else:
    state_ids = session.states[start_indices]

# %%
state_len_df = pd.DataFrame(zip(state_ids, state_lengths), columns = ('state', 'length'))

state_len_df['length'] = state_len_df['length'] * bin_length / 1000

# %%
estimator_fun = np.median

fig, ax = plt.subplots(figsize = (cb_width, cb_height / 3))

sns.stripplot(x = 'state', y = 'length', data = state_len_df, zorder = 1, hue = 'state', palette = state_colors, ax = ax, alpha = 0.3)
sns.pointplot(x = 'state', y = 'length', data = state_len_df, zorder = 100, join = None, color = 'black', ax = ax, estimator = estimator_fun, ci = None)

#ax.set_yscale('log')

ax.set_ylabel('segment length (sec.)')
ax.get_legend().remove()

sns.despine(ax = ax)

# %% [markdown]
# Behavioral states

# %%
transition_indices = get_transition_indices(session.behavioral_states)
start_indices = np.insert(transition_indices, 0, 0)

# might have to be size-1. probably not
state_lengths = np.diff(np.append(start_indices, session.behavioral_states.size))

# states at the start of each segment
if isinstance(session.behavioral_states, xr.DataArray):
    state_ids = session.behavioral_states.values[start_indices]
else:
    state_ids = session.behavioral_states[start_indices]

# %%
beh_state_len_df = pd.DataFrame(zip(state_ids, state_lengths), columns = ('state', 'length'))

beh_state_len_df['length'] = beh_state_len_df['length'] * bin_length / 1000

# %%
from hidenseek.globals import inverse_behavioral_state_dict, behavioral_state_plot_dict

# %%
beh_state_len_df = beh_state_len_df.replace({'state' : inverse_behavioral_state_dict})
beh_state_len_df = beh_state_len_df.replace({'state' : behavioral_state_plot_dict})

# %%
estimator_fun = np.median

fig, ax = plt.subplots(figsize = (cb_width, cb_height / 3))

sns.stripplot(x = 'state', y = 'length', data = beh_state_len_df, zorder = 1, ax = ax, alpha = 0.3, color = "tab:blue")
sns.pointplot(x = 'state', y = 'length', data = beh_state_len_df, zorder = 100, join = None, color = 'black', ax = ax, estimator = estimator_fun, ci = None)

ax.set_ylabel('segment length (sec.)')

plt.setp(ax.get_xticklabels(), rotation = 90)

sns.despine(ax = ax)


# %%
def _plot_segment_lengths(state_len_df, ax, estimator_fun=np.median):
    sns.stripplot(x = 'state', y = 'length', data = state_len_df, zorder = 1, hue = 'state', palette = state_colors, ax = ax, alpha = 0.3)
    sns.pointplot(x = 'state', y = 'length', data = state_len_df, zorder = 100, join = None, color = 'black', ax = ax, estimator = estimator_fun, ci = None, scale = 0.5)

    ax.set_ylabel('segment length (sec.)')
    ax.get_legend().remove()

    sns.despine(ax = ax)


# %%
def _color_state_xticklabels(ax):
    [t.set_color(tab20(norm(i))) for (i, t) in zip(range(K), ax.xaxis.get_ticklabels())];
    [t.set_fontweight('bold') for t in ax.xaxis.get_ticklabels()];


# %% [markdown]
# # Make figure 

# %%
fig = plt.figure(figsize = (cb_width, 0.7 * cb_height))

gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [1, 1], hspace = 0.3)

gs_per_state = gs[0].subgridspec(nrows = 2, ncols = 2, wspace = 0.4, hspace = 0.4)

ax_segment_lengths             = fig.add_subplot(gs_per_state[0, 0])
ax_role_given_state            = fig.add_subplot(gs_per_state[0, 1])
ax_role_given_state_normalized = fig.add_subplot(gs_per_state[1, 0])
ax_state_given_role            = fig.add_subplot(gs_per_state[1, 1])
ax_segment_lengths_label             = fig.add_subplot(gs_per_state[0, 0], xticks = [], yticks = [], frame_on = False, label = 'segments_label')
ax_role_given_state_label            = fig.add_subplot(gs_per_state[0, 1], xticks = [], yticks = [], frame_on = False, label = 'label1')
ax_role_given_state_normalized_label = fig.add_subplot(gs_per_state[1, 0], xticks = [], yticks = [], frame_on = False, label = 'label1')
ax_state_given_role_label            = fig.add_subplot(gs_per_state[1, 1], xticks = [], yticks = [], frame_on = False, label = 'label2')

_plot_segment_lengths(state_len_df, ax_segment_lengths, estimator_fun=estimator_fun)
_color_state_xticklabels(ax_segment_lengths)

sns.barplot(x = 'state', hue = 'role', y = 'P_role_given_state', data = pdf, ax = ax_role_given_state,
            palette = {'seek' : seek_color, 'hide' : hide_color})
_color_state_xticklabels(ax_role_given_state)
sns.despine(ax = ax_role_given_state)
ax_role_given_state.set_ylabel('P(role | state)')
ax_role_given_state.legend(loc = "center", bbox_to_anchor = (0.5, 1.1), fancybox = False, framealpha = 0, ncol = 2)

sns.barplot(x = 'state', hue = 'role', y = 'P_role_given_state_normalized', data = pdf, ax = ax_role_given_state_normalized,
            palette = {'seek' : seek_color, 'hide' : hide_color})
_color_state_xticklabels(ax_role_given_state_normalized)
sns.despine(ax = ax_role_given_state_normalized)
ax_role_given_state_normalized.set_ylabel('P(role | state) / P(role)')
ax_role_given_state_normalized.get_legend().remove()

sns.barplot(x = 'state', hue = 'role', y = 'P_state_given_role', data = pdf, ax = ax_state_given_role,
            palette = {'seek' : seek_color, 'hide' : hide_color})

_color_state_xticklabels(ax_state_given_role)
sns.despine(ax = ax_state_given_role)
ax_state_given_role.set_ylabel('P(state | role)')
ax_state_given_role.get_legend().remove()


gs_umap = gs[1].subgridspec(nrows = 2, ncols = 1, height_ratios = [1, 2], hspace = -0.2)
gs_upper = gs_umap[0].subgridspec(nrows = 1, ncols = 2, width_ratios = [1, 1], hspace = 0)
ax_seek = fig.add_subplot(gs_upper[0], projection='3d')
ax_hide = fig.add_subplot(gs_upper[1], projection='3d')
ax_both = fig.add_subplot(gs_umap[1], projection='3d')
ax_umap_label = fig.add_subplot(gs_umap[:], xticks = [], yticks = [], frame_on = False)

make_fig_umap_3d_hide_and_seek(session, ax_seek, ax_hide)
make_fig_umap_3d_hide_and_seek_single_axis(session, ax_both, 0.2, s = 2)

ax_seek.set_title('seek')
ax_hide.set_title('hide')
ax_both.set_title('both roles')

ax_both.get_legend().remove()

label_subfigures([ax_segment_lengths_label, ax_role_given_state_label, ax_role_given_state_normalized_label, ax_state_given_role_label, ax_umap_label], 'auto', -0.12)

fig.align_ylabels([ax_segment_lengths, ax_role_given_state_normalized])
fig.align_ylabels([ax_segment_lengths_label, ax_role_given_state_normalized_label, ax_umap_label])

#fig.tight_layout()

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')

fig.savefig(os.path.join(figures_root_dir, f'supp_fig_1_with_normalized_and_segment_lengths_K_{K}.pdf'), dpi = 400)
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_1_with_normalized_and_segment_lengths_K_{K}.png'), dpi = 400)

# %%
