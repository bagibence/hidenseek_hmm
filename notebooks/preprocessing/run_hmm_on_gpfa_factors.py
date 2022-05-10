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

# %%
from hidenseek.db_interface import *

# %%
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
import ssm
import autograd.numpy.random as npr

# %%
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed

# %%
import pickle

# %% [markdown]
# # Set parameters 

# %%
transitions = 'sticky'
K = 11

# %%
n_jobs = 20

n_repeats = 40
n_fit_iters = 5000
fit_tolerance = 1e-6

# %%
gpfa_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), 'gpfa')

hmm_results_dir = os.path.join(gpfa_dir, f'hmm_{transitions}_{K}_{n_repeats}_seeds')

if not os.path.exists(hmm_results_dir):
    os.mkdir(hmm_results_dir)

# %% [markdown]
# # Run for every session 

# %% tags=[]
results = []
for session in tqdm(Session.select()):
    gpfa_path = os.path.join(gpfa_dir, f'session_{session.id}.pickle')

    with open(gpfa_path, 'rb') as f:
        gpfa_dict = pickle.load(f)

    factors = gpfa_dict['factors']
    gpfa = gpfa_dict['gpfa']
    D = gpfa_dict['D']
    bin_length = gpfa_dict['bin_length']

    assert gpfa.x_dim == D

    y = [f.transpose('time', 'factor').values for f in factors]

    def _(i):
        # set separate random seed for each initialization
        np.random.seed(i)
        npr.seed(i)
        hmm = ssm.HMM(K=K, D=D, transitions=transitions, observations = 'gaussian')
        hmm.initialize(y)
        _ = hmm.fit(y, num_iters=n_fit_iters, tolerance=fit_tolerance)
        
        hmm.seed = i
        
        return hmm
        
    hmms = Parallel(n_jobs = n_jobs, verbose=10)(delayed(_)(i) for i in range(n_repeats))
    lls = [hmm.log_likelihood(y) for hmm in hmms]
    hmm = hmms[np.argmax(lls)]

    states = [xr.DataArray(hmm.most_likely_states(f.transpose('time', 'factor').values), dims = ('time'), coords = {'time' : f.time}) for f in factors]
    state_probs = [xr.DataArray(hmm.expected_states(f.transpose('time', 'factor').values)[0], dims = ('time', 'state'), coords = {'time' : f.time}) for f in factors]

    results.append((session.id, states, state_probs, hmm, hmm.seed))
    
for r in results:
    sid, states, state_probs, model, seed = r
    
    with open(os.path.join(hmm_results_dir, f'session_{sid}.pickle'), 'wb') as f:
        save_dict = {'K' : K,
                     'transitions' : transitions,
                     'states' : states,
                     'state_probs' : state_probs,
                     'session_id' : sid,
                     'seed' : seed,
                     'model' : model}
        pickle.dump(save_dict, f)

# %% [markdown]
# # Look at the results
# %%
try:
    results
except:
    for fname in os.listdir(hmm_results_dir):
        with open(os.path.join(hmm_results_dir, fname), 'rb') as f:
            res = pickle.load(f)
            
        session = Session[res['session_id']]
        for (trial, states) in zip(session.trials, res['states']):
            trial.states = states
else:
    for r in results:
        sid, states, state_probs, model, seed = r
        session = Session[sid]

        for (st, stp, trial) in zip(states, state_probs, session.trials):
            trial.states = st
            trial.state_probs = stp

# %%
session = Session[13]

bin_length = 250

ss_seek, ss_hide = session.get_stretched_states(bin_length)

# %%
from hidenseek.util.plotting import *

fig, ax = plt.subplots(figsize = (30, 10), nrows = 2)

plot_states_xr(ax[0], ss_seek, K)
plot_states_xr(ax[1], ss_hide, K)

add_vertical_lines_for_time_points(ax[0], session.get_median_time_points('seek') / bin_length, 'black')
add_vertical_lines_for_time_points(ax[1], session.get_median_time_points('hide') / bin_length, 'black')


# %%
for session in tqdm(Session.select()):
    gpfa_path = os.path.join(gpfa_dir, f'session_{session.id}.pickle')

    with open(gpfa_path, 'rb') as f:
        gpfa_dict = pickle.load(f)

    session.gpfa = gpfa_dict['gpfa']
    for (trial, f) in zip(session.trials, gpfa_dict['factors']):
        trial.factors = f

# %%
plt.scatter([session.recorded_cells.count() for session in Session.select()], [session.seek_trials[0].factors.factor.size for session in Session.select()])

# %%
Session[13].seek_trials[0].factors.plot()

# %%
session = Session[13]

# %%
ll = np.array(session.gpfa.fit_info['log_likelihoods'])

# %%
plt.plot(ll[np.isfinite(ll)][-100:])

# %%
plt.plot(ll[np.isfinite(ll)])

# %%
