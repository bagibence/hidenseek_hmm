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
# Training an HMM on 9/10 of the trials, then calculating the MI with behavior on the remaining trials.
#
# Generates data for Supplementary Figure 2.

# %% [markdown]
# # Imports

# %%
# NOTE: might want to use threadpoolctl for this in the future
import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
from joblib import Parallel, delayed

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.model_selection import KFold

from hidenseek.figure_util.fake_state_functions import *

# %% [markdown]
# # Load GPFA factors

# %%
from hidenseek.figure_util.load_results import load_factors

load_factors()

# %% [markdown]
# # Add behavioral states 

# %%
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

add_behavioral_states()

# %% [markdown]
# # Set parameters

# %%
K = 11

n_repeats = 10
n_folds = 10
N_fake = 10000

seed = 123

# %% [markdown]
# # All sessions

# %% tags=[]
score_tuples = []
for session in tqdm(Session.select()):
    D = session.seek_trials[0].factors.factor.size

    n_trials = len(session.trials)

    for (fold, (train_ind, test_ind)) in tqdm(enumerate(KFold(n_folds, shuffle=True).split(np.arange(n_trials))), total=n_folds):
        # train on training trials
        train_data = [trial.factors.values.T for (i, trial) in enumerate(session.trials) if i in train_ind]
        def _(i):
            np.random.seed(i)
            npr.seed(i)
            
            hmm = ssm.HMM(K=K, D=D, transitions='sticky', observations = 'gaussian')
            hmm.initialize(train_data)
            hmm.fit(train_data, num_iters=1000, tolerance=1e-6, verbose=0)
            
            return (hmm, hmm.log_likelihood(train_data))

        res = Parallel(n_jobs = n_repeats)(delayed(_)(i) for i in range(n_repeats))
        
        lls_list = [r[1] for r in res]
        hmms = [r[0] for r in res]
        hmm = hmms[np.argmax(lls_list)]

        # test on test trials
        test_states_list = [hmm.most_likely_states(trial.factors.values.T) for (i, trial) in enumerate(session.trials) if i in test_ind]
        test_behavioral_states = np.concatenate([trial.behavioral_states for (i, trial) in enumerate(session.trials) if i in test_ind])

        score_tuples.append((session.id, mutual_info_score(test_behavioral_states, np.concatenate(test_states_list)), 'real', 'mi', 0, fold))
        score_tuples.append((session.id, adjusted_mutual_info_score(test_behavioral_states, np.concatenate(test_states_list)), 'real', 'adjusted', 0, fold))

        # test with fake states for the test trials
        fake_states_list = [generate_fake_states_by_rearranging_states(test_states, N_fake, seed) for test_states in test_states_list]
        for i in range(N_fake):
            score_tuples.append((session.id, mutual_info_score(test_behavioral_states, np.concatenate([fs[i] for fs in fake_states_list])), 'fake', 'mi', i, fold))
            score_tuples.append((session.id, adjusted_mutual_info_score(test_behavioral_states, np.concatenate([fs[i] for fs in fake_states_list])), 'fake', 'adjusted', i, fold))

# %%
score_df = pd.DataFrame(score_tuples, columns = ['session', 'score', 'real', 'score_type', 'i', 'fold'])

# %%
score_df.to_csv(f'train_on_one_part_then_MI_on_other_K_{K}.csv', index = False)

# %%
for (sid, sdf) in score_df.groupby('session'):
    fig, ax = plt.subplots(figsize = (10, 4))
    sns.violinplot(x = 'fold', y = 'score', hue = 'real', data = sdf.query('score_type == "mi" and real == "fake"'), label = 'fake', ax = ax)
    sns.pointplot(x = 'fold', y = 'score', hue = 'real', data = sdf.query('score_type == "mi" and real == "real"'), palette = ['tab:orange', ], join = False, label = 'real', ax = ax)
    ax.legend();
