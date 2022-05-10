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

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
import neo
from quantities import ms
from elephant.gpfa import GPFA
from elephant.gpfa import GPFA, gpfa_core, gpfa_util

# %%
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import FactorAnalysis

# %%
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# %%
import pickle

# %% [markdown]
# # Set a global random seed 

# %%
seed = 123
np.random.seed(seed)


# %% [markdown]
# # Determine dimensionalities and run GPFA

# %%
def opt_dimensionality(FA, threshold=0.95):
    """
    Number of eigenvalues needed to explain the variance threshold
    """
    L = FA.components_
    
    evals = np.real(np.linalg.svd(L.T @ L, compute_uv=False))
    dim_ind = np.nonzero((np.cumsum(evals) / np.sum(evals)) > threshold)[0][0]
    
    return dim_ind + 1


# %%
def determine_dimensionality(session, n_FA_CV_folds, n_CV_jobs):
    """
    Extract all spikes from a session and use FA to determine dimensionality
    """
    # create square-rooted spike counts in the whole session for FA
    session_spikes = xr.concat([preproc.create_spike_train(neur.all_spikes, session.last_trial_end, bin_length) for neur in session.recorded_cells], dim = 'neuron')
    X = np.sqrt(session_spikes.transpose('time', 'neuron').values)

    # find the best FA model for cross-validated log-likelihood
    np.random.seed(seed)
    gridsearch = GridSearchCV(FactorAnalysis(),
                              {'n_components' : range(1, session.recorded_cells.count())},
                              cv = n_FA_CV_folds, verbose = False, n_jobs = n_CV_jobs)
    gridsearch.fit(X)

    # determined the population dimensionality like Semedo
    D = opt_dimensionality(gridsearch.best_estimator_, FA_threshold)
    
    return D


# %% tags=[]
def get_lls(gpfa):
    return list(filter(np.isfinite, gpfa.fit_info['log_likelihoods']))


# %%
bin_length = 250

FA_threshold = 0.95

n_FA_CV_folds = 10
n_CV_jobs = 10

max_gpfa_iters = 5000
n_repeats = 10

# %%
if not os.path.exists(os.getenv('PROCESSED_DATA_DIR')):
    os.mkdir(os.getenv('PROCESSED_DATA_DIR'))
    
results_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), f'gpfa')

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# %%
exec_order = [Session[13], Session[12], *[s for s in Session.select() if s.id not in [12, 13]]]

# %% tags=[] jupyter={"outputs_hidden": true}
for session in tqdm(exec_order):
    
    print('Determining population dimensionality using FA')
    D = determine_dimensionality(session, n_FA_CV_folds, n_CV_jobs)
    session.D = D
    print(f'Found dimensionality of session {session.id}: {session.recorded_cells.count()} -> {D}')

    # create spike trains to feed to GPFA 
    trial_trains = [[neo.SpikeTrain(st, trial.time_points.end, units = ms) for st in trial.spike_times]
                    for trial in session.trials]

    # fit GPFA and get the factors
    def _(i):
        np.random.seed(i)
        gpfa = GPFA(bin_length * ms, D, verbose=False, em_max_iters=max_gpfa_iters)
        gpfa.fit(trial_trains)
        gpfa.random_seed = i
        return gpfa

    models = Parallel(n_jobs = n_repeats, verbose = 10)(delayed(_)(i) for i in range(n_repeats))

    session.best_gpfa_model = models[np.argmax([get_lls(m)[-1] for m in models])]
    session.trial_trains = trial_trains


# %% tags=[]
def project_and_save(sid, model, trial_trains):
    try:
        connect_to_db()
    except pony.orm.core.BindingError:
        pass
    
    with db_session:
        session = Session[sid]
        
        results = model.transform(trial_trains,
                                  returned_data=['latent_variable_orth',
                                                 'latent_variable'])
        x_orth = results['latent_variable_orth']
        x_orig = results['latent_variable']

        factors_xr_list = [xr.DataArray(x, dims = ['factor', 'time'], coords = {'time' : trial.get_spike_trains(bin_length).time[:-1]})
                           for (trial, x) in zip(session.trials, x_orth)]
        factors_xr_list_orig = [xr.DataArray(x, dims = ['factor', 'time'], coords = {'time' : trial.get_spike_trains(bin_length).time[:-1]})
                                for (trial, x) in zip(session.trials, x_orig)]
        
        out_filename = os.path.join(results_dir, f'session_{session.id}.pickle')
        with open(out_filename, 'wb') as f:
            pickle.dump({'bin_length' : bin_length,
                         'session_id' : session.id,
                         'FA_threshold' : FA_threshold,
                         'D' : model.get_params()['x_dim'],
                         'gpfa' : model,
                         'seed' : model.random_seed,
                         'factors' : factors_xr_list,
                         'factors_orig' : factors_xr_list_orig,
                         'description' : f'GPFA was fit on square-rooted spike counts with D determined by cross-validated FA with FA_threshold for the optimal dimensionality. factors are the orhogonalized and ordered factors (use these for further analysis), factors_orig the original factors'},
                        f)
        
Parallel(n_jobs = 13)(delayed(project_and_save)(s.id, s.best_gpfa_model, s.trial_trains) for s in Session.select())

# %%
for session in Session.select():
    out_filename = os.path.join(results_dir, f'session_{session.id}.pickle')
    with open(out_filename, 'rb') as f:
        results = pickle.load(f)
        
    session.gpfa = results['gpfa']
    #for (t, f) in zip(session.trials, results['factors']):
    #    t.factors = f

# %%
for session in Session.select():
    ll = get_lls(session.gpfa)

    plt.figure()
    plt.plot(ll[-100:])
    plt.title(f'session {session.id}, {session.recorded_cells.count()} neurons')

# %%
