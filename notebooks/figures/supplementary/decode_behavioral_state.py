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
# Supervised decoding of the tagged behavioral states from the GPFA factors.
#
# Generates data plotted in Supplementary Figure 2.

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

import os
import pickle
from joblib import Parallel, delayed

import ssm
import autograd.numpy.random as npr

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')

# %%
from hidenseek.figure_util.load_results import load_results, load_factors

K = 11
transitions = 'sticky'

bin_length = load_results(K, transitions)

# %%
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

add_behavioral_states()

# %%
for session in tqdm(Session.select()):
    session.factors = np.row_stack([trial.factors.values.T for trial in session.trials])
    session.hmm_states = np.concatenate([trial.states.values for trial in session.trials])
    session.behavioral_states = np.concatenate([trial.behavioral_states.values for trial in session.trials])

# %% [markdown]
# # Decode from the factors

# %%
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score

# %% [markdown]
# ## CV trial-wise 
#
# Separate training and test data on the level of trials, so that it cannot happen that time points from the same trial are both in the test and training sets.
#
# In each iteration, leave one trial as the test set and train on the rest.

# %%
np.random.seed(123)

# %% tags=[]
cv = KFold(10, shuffle=True, random_state=123)
#cv = LeaveOneOut()

cv_tuples = []

for session in tqdm(Session.select()):
    n_trials = len(session.trials)

    for (fold, (train_ind, test_ind)) in enumerate(cv.split(np.arange(n_trials))):
        train_trials = [trial for (i, trial) in enumerate(session.trials) if i in train_ind]
        test_trials = [trial for (i, trial) in enumerate(session.trials) if i in test_ind]

        train_factors = np.row_stack([trial.factors.values.T for trial in train_trials])
        test_factors = np.row_stack([trial.factors.values.T for trial in test_trials])

        train_behavioral_states = np.concatenate([trial.behavioral_states for trial in train_trials])
        test_behavioral_states = np.concatenate([trial.behavioral_states for trial in test_trials])

        lda = LinearDiscriminantAnalysis().fit(train_factors, train_behavioral_states)
        nb = GaussianNB().fit(train_factors, train_behavioral_states)
        rf = RandomForestClassifier().fit(train_factors, train_behavioral_states)
        dummy_stratified = DummyClassifier(strategy='stratified').fit(train_factors, train_behavioral_states)
        dummy_most_frequent = DummyClassifier(strategy='most_frequent').fit(train_factors, train_behavioral_states)

        cv_tuples.append((session.id, 'lda', len(session.recorded_cells), lda.score(test_factors, test_behavioral_states), fold))
        cv_tuples.append((session.id, 'naive_bayes', len(session.recorded_cells), nb.score(test_factors, test_behavioral_states), fold))
        cv_tuples.append((session.id, 'random_forest', len(session.recorded_cells), rf.score(test_factors, test_behavioral_states), fold))
        cv_tuples.append((session.id, 'dummy_stratified', len(session.recorded_cells), dummy_stratified.score(test_factors, test_behavioral_states), fold))
        cv_tuples.append((session.id, 'dummy_most_frequent', len(session.recorded_cells), dummy_most_frequent.score(test_factors, test_behavioral_states), fold))

# %%
cv_df = pd.DataFrame(cv_tuples, columns = ['session', 'method', 'num_cells', 'score', 'fold'])

# %%
if isinstance(cv, KFold):
    cv_type = f'cv_{cv.n_splits}'
else:
    cv_type = 'leave_one_out'

cv_df.to_csv(f'decode_behavior_cv_results_K_{K}_{cv_type}.csv')

# %% [markdown]
# Plot performance

# %%
sns.catplot(x = 'session', y = 'score', col = 'method', data = cv_df, kind = 'swarm')
sns.catplot(x = 'session', y = 'score', col = 'method', data = cv_df, kind = 'point')

# %%
