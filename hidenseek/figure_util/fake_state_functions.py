import numpy as np
import xarray as xr
import pandas as pd

from tqdm.auto import tqdm

from hidenseek.util.postproc.transitions import get_transition_indices

from hidenseek.db_interface import *


def generate_fake_states_by_sampling_labels(real_states, K, N, seed):
    """
    Generate fake states by sampling a new state label from the K available labels,
    keeping transition points intact.
    
    Parameters
    ----------
    real_states : array-like (or xr.DataArray?)
        inferred states on a trial
    N : int, default 1
        number of "fake" states to generate
    seed : int
        random seed to use
        
    Returns
    -------
    list of arrays if N > 1
    a single array if N == 1
    """
    np.random.seed(seed)

    transition_indices = get_transition_indices(real_states)
    start_indices = np.insert(transition_indices, 0, 0)

    # might have to be size-1. probably not
    state_lengths = np.diff(np.append(start_indices, real_states.size))

    # states at the start of each segment
    if isinstance(real_states, xr.DataArray):
        state_ids = real_states.values[start_indices]
    else:
        state_ids = real_states[start_indices]

    shuffled_states_list = []
    for i in range(N):
        shuffled_state_ids = np.random.choice(K, size = len(state_ids)) # not using the trial's states but randomly sampling from the K possible states

        shuffled_states = np.concatenate([sid * np.ones(l) for (sid, l) in zip(shuffled_state_ids, state_lengths)])
        shuffled_states_list.append(shuffled_states)
        
    if N == 1:
        return shuffled_states_list[0]
    else:
        return shuffled_states_list


def generate_fake_states_by_shuffling_labels(real_states, N, seed):
    """
    Generate fake states by shuffling the state labels, keeping transition points intact.
    
    Parameters
    ----------
    real_states : array-like (or xr.DataArray?)
        inferred states on a trial
    N : int, default 1
        number of "fake" states to generate
    seed : int
        random seed to use
        
    Returns
    -------
    list of arrays if N > 1
    a single array if N == 1
    """
    np.random.seed(seed)

    transition_indices = get_transition_indices(real_states)
    start_indices = np.insert(transition_indices, 0, 0)

    # might have to be size-1. probably not
    state_lengths = np.diff(np.append(start_indices, real_states.size))

    # states at the start of each segment
    if isinstance(real_states, xr.DataArray):
        state_ids = real_states.values[start_indices]
    else:
        state_ids = real_states[start_indices]

    shuffled_states_list = []
    for i in range(N):
        shuffled_state_ids = np.random.permutation(state_ids) # shuffle the trial's states' order

        shuffled_states = np.concatenate([sid * np.ones(l) for (sid, l) in zip(shuffled_state_ids, state_lengths)])
        shuffled_states_list.append(shuffled_states)
        
    if N == 1:
        return shuffled_states_list[0]
    else:
        return shuffled_states_list


def generate_fake_states_by_rearranging_states(real_states, N, seed):
    """
    Generate fake states by taking the segments and their associated inferred state and shuffle them in time.
    
    Parameters
    ----------
    real_states : array-like (or xr.DataArray?)
        inferred states on a trial
    N : int, default 1
        number of "fake" states to generate
    seed : int
        random seed to use
        
    Returns
    -------
    list of arrays if N > 1
    a single array if N == 1
    """
    np.random.seed(seed)

    transition_indices = get_transition_indices(real_states)
    start_indices = np.insert(transition_indices, 0, 0)

    # might have to be size-1. probably not
    state_lengths = np.diff(np.append(start_indices, real_states.size))

    # states at the start of each segment
    if isinstance(real_states, xr.DataArray):
        state_ids = real_states.values[start_indices]
    else:
        state_ids = real_states[start_indices]

    shuffled_states_list = []
    for i in range(N):
        rand_order = np.random.permutation(len(state_lengths))
        shuffled_state_lengths = state_lengths[rand_order]
        shuffled_state_ids = state_ids[rand_order]

        shuffled_states = np.concatenate([sid * np.ones(l) for (sid, l) in zip(shuffled_state_ids, shuffled_state_lengths)])
        shuffled_states_list.append(shuffled_states)
        
    if N == 1:
        return shuffled_states_list[0]
    else:
        return shuffled_states_list


def generate_fake_states(method_fun, N, seed):
    """
    Generate N fake state sequences using method_fun

    Parameters
    ----------
    method_fun : function
        function that takes real states and generates fake states
    N : int, default 1
        number of "fake" states to generate
    seed : int
        random seed to use

    Returns
    -------
    None, but adds .fake_states_list to all trials and all sessions
    """
    # generate fake states and add them to the trial objects
    for trial in tqdm(Trial.select()):
        trial.fake_states_list = method_fun(trial.states, N, seed)
     
    # concatenate fake states in the session's trials
    for session in tqdm(Session.select()):
        session.fake_states_list = [np.concatenate([trial.fake_states_list[i] for trial in session.trials])
                                    for i in range(N)]
        

def add_real_and_fake_scores_to_sessions(score_fun, score_kwargs):
    """
    Calculate MI scores with behavior for the HMM states and the fake states
    for all sessions

    Parameters
    ----------
    score_fun : function
        function that calculates the MI
    score_kwargs : dict
        keyword arguments to pass to score_fun

    Returns
    -------
    None, but adds .real_score and .fake_scores to each session
    """
    # calculate real and fake scores session-wise
    for session in tqdm(Session.select()):
        session.real_score = score_fun(session.behavioral_states,
                                       session.states,
                                       **score_kwargs)

        session.fake_scores = [score_fun(session.behavioral_states,
                                         fake_states,
                                         **score_kwargs)
                               for fake_states in session.fake_states_list]
        
        
def get_scores_df():
    """
    Collect real and fake MI scores from all sessions into a single dataframe

    Returns
    -------
    sessionwise_scores_df : pd.DataFrame
        dataframe containing the real and fake scores
        columns: session_id, score, kind
        kind can be "real" or "fake"
    """
    # create dataframe for plotting
    sessionwise_real_scores_df = pd.DataFrame([(session.id, session.real_score) for session in Session.select()], columns = ['session_id', 'score'])
    sessionwise_real_scores_df['kind'] = 'real'

    sessionwise_fake_scores_df = pd.DataFrame([(session.id, s) for session in Session.select()
                                                               for s in session.fake_scores],
                                              columns = ['session_id', 'score'])
    sessionwise_fake_scores_df['kind'] = 'fake'

    sessionwise_scores_df = pd.merge(sessionwise_real_scores_df, sessionwise_fake_scores_df, how = 'outer')
    
    return sessionwise_scores_df


def circular_shuffle_states(session, by_whole_segments=False):
    """
    Perform all possible circular permutations of a session's states

    Parameters
    ----------
    session : Session
        session whose states to shift
    by_whole_segments : bool, default False
        True: shift by whole HMM segments
        False: shift by every possible number of timepoints

    Returns
    -------
    shifted_states_list : list of arrays
        list of shifted state sequences
    """
    session_states = session.states.values

    transition_indices = get_transition_indices(session_states)
    start_indices = np.insert(transition_indices, 0, 0)

    shifted_states_list = []
    if by_whole_segments:
        for n_segments_to_shift_by in range(1, len(transition_indices)):
            len_segment_to_shift_by = start_indices[n_segments_to_shift_by]
        
            shifted_states = np.roll(session_states, len_segment_to_shift_by)

            shifted_states_list.append(shifted_states)
    else:
        for len_segment_to_shift_by in range(1, len(session_states)):
            shifted_states = np.roll(session_states, len_segment_to_shift_by)

            shifted_states_list.append(shifted_states)
        
    return shifted_states_list
