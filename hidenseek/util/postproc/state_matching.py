import numpy as np
import pandas as pd
import xarray as xr

from scipy.optimize import linear_sum_assignment

from hidenseek.db_interface import *

from tqdm.auto import tqdm

def pair_states(correlations, method):
    """
    Pair the states corresponding to each other based on the cross-correlation matrix 
    
    Parameters
    ----------
    correlations : ?
        cross-correlation matrix of the states of two sessions
    method : str
        method to use for assigning the pairs
        'own' : use my method
        'hungarian' : use the Hungarian algorithm from scipy.optimize
    
    Returns
    -------
    pairs : np.array (#states x 2)
        pairs of states that are most correlated to each other
    pair_correlations : list
        correlations of the state pairs
    """
    assert correlations.shape[0] == correlations.shape[1]
    cm = correlations.copy()
    
    if method == 'own':
        pairs = np.full((cm.shape[0], 2), None)
        pair_correlations = []
        num_pairs = 0
        while (pairs == None).sum() > 0:
            # highest correlations
            s1, s2 = np.unravel_index(np.argmax(cm), cm.shape)

            pair_corr = cm[s1, s2].copy()
            cm[s1, s2] = -1

            if s1 not in pairs[:, 0] and s2 not in pairs[:, 1]:
                pairs[num_pairs, :] = (s1, s2)
                num_pairs += 1
                pair_correlations.append(pair_corr)

        sort_ind = np.argsort(pairs[:, 0])
        pairs = pairs[sort_ind].astype(int)
    
    elif method == 'hungarian':
        cm[np.isnan(cm)] = -1
        row_ind, col_ind = linear_sum_assignment(cm, maximize = True)
        pairs = np.array([(a, b) for (a, b) in zip(row_ind, col_ind)])
        pair_correlations = [cm[a, b] for (a, b) in pairs]
        
    return pairs, pair_correlations


def get_state_correlations(K, method, smoothing_len=5):
    """
    Calculate the correlations between the states of all the sessions
    based on the state_hist_* attributes of the sessions

    Parameters
    ----------
    K : int
        number of different possible states
    method : str
        method to use for matching the states of different sessions
        see pair_states
    smoothing_len : int, default 5
        number of time points to use for a rolling average
        when smoothing the states for a nicer correlation

    Returns
    -------
    session_match_df : pd.DataFrame
        DataFrame with columns (session_a, session_b, state_a, state_b, correlation, role)
    """
    # calculate correlations between the states of different sessions

    session_match_tuples = [] 
    for session_a in Session.select():
        for session_b in Session.select():
            seek_corrs = np.corrcoef(session_a.state_hist_seek.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_seek.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
            hide_corrs = np.corrcoef(session_a.state_hist_hide.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_hide.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
            seek_pairs, pair_corrs_seek = pair_states(seek_corrs, method)
            hide_pairs, pair_corrs_hide = pair_states(hide_corrs, method)
            
            for seek_pair, pair_corr in zip(seek_pairs, pair_corrs_seek):
                session_match_tuples.append((session_a.id, session_b.id, *seek_pair, pair_corr, 'seek')) 
            for hide_pair, pair_corr in zip(hide_pairs, pair_corrs_hide):
                session_match_tuples.append((session_a.id, session_b.id, *hide_pair, pair_corr, 'hide'))

    session_match_df = pd.DataFrame(session_match_tuples, columns = ('session_a', 'session_b', 'state_a', 'state_b', 'correlation', 'role'))
    return session_match_df


def match_states_to_reference(ref_session, smoothing_len, use_most_likely, matching_method,
                              median_time_points_seek=None, median_time_points_hide=None, save_orig=True):
    """
    Match the states of all the sessions to a chosen reference session

    Parameters
    ----------
    ref_session : Session
        reference session
    smoothing_len : int
        length of the moving average window
    use_most_likely : bool
        True: use probabilities extracted based on the most likely state at every time point
        False: use probabilities from the HMM directly
    matching_method : str
        method to use for pairing the states. gets passed to pair_states
        'own' or 'hungarian'
    median_time_points_seek : pd.Series
        median time points across all sessions that we used to stretch every trial to the same length in seek
    median_time_points_hide : pd.Series
        median time points across all sessions that we used to stretch every trial to the same length in hide
    save_orig : bool, default True
        save original states of the trials in trial.orig_states
        otherwise they are lost because the matching overwrites trial.states

    Returns
    -------
    reference_correlations : dict
        keys are the IDs of the sessions
        values are the correlations of the states to their pair in the reference session
    """
    K = len(ref_session.state_hist_seek.state)
    bin_length = int(np.diff(ref_session.seek_trials[0].states.time[:2]))
    reference_correlations = {}

    for session_b in tqdm(Session.select()):
        if use_most_likely:
            seek_corrs = np.corrcoef(ref_session.state_hist_seek.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_seek.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
            hide_corrs = np.corrcoef(ref_session.state_hist_hide.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_hide.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
        else:
            seek_corrs = np.corrcoef(ref_session.state_hist_seek_dir.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_seek_dir.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
            hide_corrs = np.corrcoef(ref_session.state_hist_hide_dir.rolling(time = smoothing_len).mean().dropna('time').values,
                                     session_b.state_hist_hide_dir.rolling(time = smoothing_len).mean().dropna('time').values)[:K, K:]
            
        
        #pairs, pair_corrs = pair_states(seek_corrs)
        corr_matrix = (seek_corrs + hide_corrs) / 2
        corr_matrix[np.isnan(corr_matrix)] = -1
        pairs, pair_corrs = pair_states(corr_matrix, matching_method)

        reference_correlations[session_b.id] = pair_corrs
    
        # save the original states in the orig_states attribute
        if save_orig:
            for trial in session_b.trials:
                trial.orig_states = trial.states.copy()
        
        # reorder the states and state probabilities of the session based on the pairings
        for trial in session_b.trials:
            # for the most likely states
            masks = [trial.states == pair[1] for pair in pairs]
            for mask, pair in zip(masks, pairs):
                trial.states[mask] = pair[0]
                
            # for the state probabilities
            trial.state_probs = trial.state_probs.sel(state = pairs[:, 1])
            
        # regenerate the histograms for the session using the reordered states in the trials
        session_b.state_hist_seek, session_b.state_hist_hide = session_b.get_state_hist(bin_length, K, median_time_points_seek, median_time_points_hide)
        session_b.state_probs_seek, session_b.state_probs_hide = session_b.get_state_probs(bin_length, median_time_points_seek, median_time_points_hide)
        session_b.state_hist_seek_dir = session_b.state_probs_seek.mean('trial')
        session_b.state_hist_hide_dir = session_b.state_probs_hide.mean('trial')
        
    return reference_correlations


