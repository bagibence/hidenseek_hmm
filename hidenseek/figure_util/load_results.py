from hidenseek.db_interface import *

import pickle
import os

def load_factors(gpfa_dir=None):
    """
    Load and add fitted GPFA factors to all trials

    Parameters
    ----------
    gpfa_dir : str, optional
        directory where the GPFA models are stored in

    Returns
    -------
    None, but adds .factors to each trial
    """
    if gpfa_dir is None:
        gpfa_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), 'gpfa')
    
    bin_length_vals = []

    for s in Session.select():
        gpfa_fname = os.path.join(gpfa_dir, f'session_{s.id}.pickle')

        with open(gpfa_fname, 'rb') as f:
            gpfa_dict = pickle.load(f)

        bin_length_vals.append(gpfa_dict['bin_length'])

        for (trial, f) in zip(s.trials, gpfa_dict['factors']):
            trial.factors = f

    assert len(set(bin_length_vals)) == 1

    bin_length = bin_length_vals[0]

    return bin_length


def load_results(K, transitions, gpfa_dir=None, n_seeds=40):
    """
    Load HMM states and state probabilities, and add them to trials

    Parameters
    ----------
    K : int
        number of states in the model to load
    transitions : str
        transition type of the model to load
    gpfa_dir : str, optional
        directory where the GPFA results to which
        the HMM was fitted are stored in
    n_seeds : int, default 40
        number of random seeds that were used during the fitting

    Returns
    -------
    (K, bin_length) : (int, int)
        K : number of states
        bin_length : bin length used during the fitting in ms
    Also adds .states and .state_probs to all trials
    """
    if gpfa_dir is None:
        gpfa_dir = os.path.join(os.getenv('PROCESSED_DATA_DIR'), 'gpfa')

    bin_length = load_factors(gpfa_dir)
    hmm_results_dir = os.path.join(gpfa_dir, f'hmm_{transitions}_{K}_{n_seeds}_seeds')

    K_vals = []

    for s in Session.select():
        hmm_fname = os.path.join(hmm_results_dir, f'session_{s.id}.pickle')

        with open(hmm_fname, 'rb') as f:
            hmm_dict = pickle.load(f)

        K_vals.append(hmm_dict['K'])

        for (trial, st, stp) in zip(s.trials, hmm_dict['states'], hmm_dict['state_probs']):
            trial.states = st
            trial.state_probs = stp

        s.states = xr.concat([t.states.drop('time') for t in s.trials], dim = 'time')

    assert len(set(K_vals)) == 1

    K = K_vals[0]

    return K, bin_length
