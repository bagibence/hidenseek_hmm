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
# This notebook generates "fake" HMM states and calculates the mutual information with behavior.
#
# The data generated is plotted in Figure 2E.

# %%
import os
from tqdm.autonotebook import tqdm

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
from hidenseek.util.plotting.plotting_setup import *

# %%
from hidenseek.figure_util.load_results import load_results

K = 11
transitions = 'sticky'

_, bin_length = load_results(K, transitions)

# %%
save_df = False

# %% [markdown] tags=[]
# # Add behavioral states

# %%
from hidenseek.globals import (
    behavioral_state_names_seek,
    behavioral_state_names_hide,
    behavioral_state_names_extra,
    behavioral_state_dict,
    inverse_behavioral_state_dict
)

from hidenseek.util.postproc import make_behavioral_states_str, convert_str_states_to_int
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

# %%
add_behavioral_states()

# %% [markdown]
# # Set parameters 

# %%
import sklearn.metrics

score_fun = sklearn.metrics.mutual_info_score
score_kwargs = {}

seed = 123
N = 10000


# %%
from hidenseek.figure_util.fake_state_functions import *

# %% [markdown]
# # Generate fake states and calculate MI scores

# %% jupyter={"outputs_hidden": true} tags=[]
generate_fake_states(generate_fake_states_by_rearranging_states, N, seed)

add_real_and_fake_scores_to_sessions(score_fun, score_kwargs)

sessionwise_scores_df = get_scores_df()

# %%
sns.violinplot(y = 'score', x = 'session_id', data = sessionwise_scores_df.query('kind == "fake"'))
sns.pointplot(y = 'score', x = 'session_id', data = sessionwise_scores_df.query('kind == "real"'), join=None)

# %%
for (sid, subdf) in sessionwise_scores_df.groupby('session_id'):
    real_score = subdf.query('kind == "real"').score.values[0]
    
    print(np.sum(subdf.query('kind == "fake"').score > real_score), end = ', ')

# %%
if save_df:
    sessionwise_scores_df.to_csv(f'real_and_fake_mi_scores_K_{K}.csv')

# %% [markdown]
# # Alternative method
#
# Generate the fake states by sampling transition points from a uniform distribution within the trial.

# %%
from hidenseek.util.postproc.transitions import generate_fake_states_with_same_number_of_transitions

# %%
for trial in tqdm(Trial.select()):
    trial.fake_states_list = generate_fake_states_with_same_number_of_transitions(trial.states, N, seed)

# %%
# concatenate fake states in the session's trials
for session in tqdm(Session.select()):
    session.fake_states_list = [relabel_partitions(np.concatenate([trial.fake_states_list[i] for trial in session.trials]))
                                for i in range(N)]

# %%
# calculate real and fake scores session-wise
for session in tqdm(Session.select()):
    relabeled_behavioral_states = relabel_partitions(session.behavioral_states)
    relabeled_hmm_states = relabel_partitions(session.states)
    
    session.relabeled_score = score_fun(relabeled_behavioral_states, relabeled_hmm_states, **score_kwargs)
    
    session.fake_scores = [score_fun(relabeled_behavioral_states, fake_states, **score_kwargs)
                           for fake_states in session.fake_states_list]

# %%
# create dataframe for plotting
sessionwise_real_scores_df = pd.DataFrame([(session.id, session.relabeled_score) for session in Session.select()], columns = ['session_id', 'score'])
sessionwise_real_scores_df['kind'] = 'real'

sessionwise_fake_scores_df = pd.DataFrame([(session.id, s) for session in Session.select()
                                                           for s in session.fake_scores],
                                          columns = ['session_id', 'score'])
sessionwise_fake_scores_df['kind'] = 'fake'

sessionwise_scores_df = pd.merge(sessionwise_real_scores_df, sessionwise_fake_scores_df, how = 'outer')


# %%
sns.violinplot(y = 'score', x = 'session_id', data = sessionwise_scores_df.query('kind == "fake"'))
sns.pointplot(y = 'score', x = 'session_id', data = sessionwise_scores_df.query('kind == "real"'), join=None)

# %%
