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
# This notebook generates Supplementary Figure 3 in which we show the number of sessions an HMM state was associated with a given tagged behavior, for each tagged behavior.

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import pickle
from joblib import Parallel, delayed

import ssm
import autograd.numpy.random as npr

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
from hidenseek.figure_util.load_results import load_results, load_factors
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states
from hidenseek.util.plotting.plotting_setup import *

from hidenseek.util.plotting import get_tab20_and_norm, label_subfigures

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')

# %%
K = 11
transitions = 'sticky'

_, bin_length = load_results(K, transitions)

tab20, norm = get_tab20_and_norm(K)

# %%
add_behavioral_states()

# %% [markdown]
# # Match HMM states to reference

# %%
from hidenseek.util.postproc import get_median_time_points_for_every_session
from hidenseek.util.postproc.state_matching import match_states_to_reference

# calculate median time points aggregating every session
median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)

# add state histograms to the sessions
for session in tqdm(Session.select()):
    session.state_hist_seek, session.state_hist_hide = session.get_state_hist(bin_length, K, median_time_points_seek_bo, median_time_points_hide)
    session.state_probs_seek, session.state_probs_hide = session.get_state_probs(bin_length, median_time_points_seek_bo, median_time_points_hide)
    session.state_hist_seek_dir = session.state_probs_seek.mean('trial')
    session.state_hist_hide_dir = session.state_probs_hide.mean('trial')
    
    session.state_hist_seek['time'], session.state_hist_hide['time'] = session.state_probs_seek.time, session.state_probs_hide.time
    
    
# match the states of the different sessions to the reference session
session_a = Session[13]
smoothing_len = 5
use_most_likely = True
matching_method = 'hungarian'

reference_correlations = match_states_to_reference(session_a, smoothing_len, use_most_likely, matching_method, median_time_points_seek_bo, median_time_points_hide, True)

# %% [markdown]
# # Generate fake HMM states 

# %%
N = 10000

seed = 123

# %%
from hidenseek.figure_util.fake_state_functions import generate_fake_states_by_rearranging_states

for session in tqdm(Session.select()):
    # add fake states to trials
    for trial in tqdm(session.trials):
        trial.fake_states_list = generate_fake_states_by_rearranging_states(trial.states, N = N, seed=seed)

    # concatenate the fake states from the session's trials
    session.fake_states_list = [np.concatenate([trial.fake_states_list[i] for trial in session.trials])
                                for i in range(N)]

# %%
from hidenseek.globals import *

from hidenseek.figure_util.contingency import make_cont_dfs, behavioral_state_plot_dict

# %%
for trial in Trial.select():
    trial.playing_states = trial.behavioral_states

# %% tags=[]
p_value = 5 / 100

correction = 'holm'

for session in tqdm(Session.select()):
    cont_df, cont_df_hmm, cont_df_beh, signif_cont_df, signif_cont_df_hmm, signif_cont_df_beh = make_cont_dfs(session,
                                                                                                              'playing',
                                                                                                              'playing',
                                                                                                              p_value,
                                                                                                              correction=correction)
    
    # 0.001 is the cutoff for the conditional probability, it's not a p-value
    session.signif_cont_xr = (signif_cont_df.astype(bool) & cont_df > 0.001).to_xarray().to_array(dim = 'hmm_state')
    session.signif_cont_xr_hmm = (signif_cont_df_hmm.astype(bool) & (cont_df_hmm > 0.001)).to_xarray().to_array(dim = 'hmm_state')
    session.signif_cont_xr_beh = (signif_cont_df_beh.astype(bool) & (cont_df_beh > 0.001)).to_xarray().to_array(dim = 'hmm_state')

# %%
sns.heatmap(xr.concat([session.signif_cont_xr_beh.sum('hmm_state') for session in Session.select()],
                      dim = 'session').to_pandas())

plt.title('number of significant HMM states per behavior and session')  

# %%
sns.heatmap(xr.concat([session.signif_cont_xr_hmm.sum('behavioral_state') for session in Session.select()],
                      dim = 'session').to_pandas())

plt.title('number of significant behavioral states per HMM state and session')  

# %%
from hidenseek.util.plotting.plotting_setup import *

# %%
fig, ax = plt.subplots(figsize = (cb_width, cb_height / 4))


# session x hmm_state x behavior  ->  xr.concat([session.signif_cont_xr_beh for session in Session.select()], dim = 'session')
# count number of significant hmm states per behavior per session  ->  session x behavior ->  xr.concat([session.signif_cont_xr_beh for session in Session.select()], dim = 'session').sum('hmm_state')
# check if that there is one or not  ->  session x behavior ->  xr.concat([session.signif_cont_xr_beh for session in Session.select()], dim = 'session').sum('hmm_state') > 0
# count the number of sessions  ->  1 x behavior ->  (xr.concat([session.signif_cont_xr_beh for session in Session.select()], dim = 'session').sum('hmm_state') > 0).sum('session')

signif_series = (xr.concat([session.signif_cont_xr_beh for session in Session.select()], dim = 'session').sum('hmm_state') > 0).sum('session').to_pandas()
signif_series.rename(behavioral_state_plot_dict).plot.bar(ax = ax, grid = True)

ax.set_ylabel(f'Number of sessions with p < {p_value}')
ax.set_xlabel('Behavioral state')
ax.set_axisbelow(True)

sns.despine(ax = ax)

# %%
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_num_signif_conditional_probs_per_behavior_K_{K}_{correction}_correction.png'), dpi = 400, bbox_inches = 'tight')
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_num_signif_conditional_probs_per_behavior_K_{K}_{correction}_correction.pdf'), dpi = 400, bbox_inches = 'tight')

# %%
