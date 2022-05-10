import numpy as np
import pandas as pd
import xarray as xr

import sklearn.metrics

from .transitions import get_transition_indices


def correct_starts_and_ends(start_times, end_times):
    """
    Correct start and end times lists so that the start times are always before the end times creating proper intervals

    Parameters
    ----------
    start_times : np.array
        start times
    end_times : np.array
        end times

    Returns
    -------
    starts, ends : tuple of np.arrays
        corrected start and end times
    """
    starts, ends = start_times.copy(), end_times.copy()
    
    if len(starts) == len(ends) != 0:
        if starts[0] > ends[0]:
            starts = np.insert(starts, 0, 0)
            ends = np.append(ends, None)
    elif len(starts) == len(ends) + 1:
        ends = np.append(ends, None)
    elif len(starts) == len(ends) - 1:
        starts = np.insert(starts, 0, 0)
    
    assert len(starts) == len(ends)

    return starts, ends


def get_state_hist(stretched_states, K, density=True):
    """
    Calculate the number of times a state was active in the
    different trials at a given timestep
    
    Parameters
    ----------
    stretched_states : xr.DataArray
        dim x time DataArray containing the active state
        for every trial and time point with the trials stretched to the same length
    K : int
        number of different states
    density : bool, default True
        whether to normalize the to a prob. distribution at every timestep

    Returns
    -------
    state_hist : xr.DataArray
        state x time DataArray
    """
    other_dim = [dim_name for dim_name in stretched_states.dims if dim_name != 'time'][0]

    states_hist_list = [np.bincount(stretched_states.sel(time=t).dropna(other_dim), minlength=K)
                        for t in stretched_states.time]
    
    state_hist = xr.DataArray(np.stack(states_hist_list), dims=['time', 'state']).T
    state_hist['time'] = stretched_states.time

    if density:
        state_hist = state_hist / state_hist.sum('state')

    return state_hist


def get_median_time_points_for_every_session(session_wise, dropna=True):
    """
    Get median time points using every session

    Parameters
    ----------
    session_wise : bool
        if True: take the medians inside sessions and then take the medians of those
        if False: take the time points from every trial of every session and
        take the median of those
    dropna : bool, default True
        disregard trials where some of the time points are NaN

    Returns
    -------
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide
        tuple of pd.Series objects with the median time points in the roles
    """
    from hidenseek.db_interface import Session

    if session_wise:
        # 1. get the median time points for every session separately
        median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = [], [], []
        for session in Session.select():
            median_time_points_seek.append(session.get_median_time_points('seek', False, dropna = dropna))
            median_time_points_seek_bo.append(session.get_median_time_points('seek', True, dropna = dropna))
            median_time_points_hide.append(session.get_median_time_points('hide', dropna = dropna))

        median_time_points_seek_df    = pd.DataFrame(median_time_points_seek)
        median_time_points_seek_bo_df = pd.DataFrame(median_time_points_seek_bo)
        median_time_points_hide_df    = pd.DataFrame(median_time_points_hide)

        # 2. calculate the median of the median time points
        return (median_time_points_seek_df.median(),
                median_time_points_seek_bo_df.median(),
                median_time_points_hide_df.median())
    else:
        time_points_seek, time_points_seek_bo, time_points_hide = [], [], []
        for session in Session.select():
            for trial in session.seek_trials:
                time_points_seek.append(trial.get_time_points(box_open = False))
                time_points_seek_bo.append(trial.get_time_points())
            for trial in session.hide_trials:
                time_points_hide.append(trial.get_time_points())

        time_points_seek_df    = pd.DataFrame(time_points_seek)
        time_points_seek_bo_df = pd.DataFrame(time_points_seek_bo)
        time_points_hide_df    = pd.DataFrame(time_points_hide)

        if dropna:
            return (time_points_seek_df.dropna().median(),
                    time_points_seek_bo_df.dropna().median(),
                    time_points_hide_df.dropna().median())
        else:
            return (time_points_seek_df.median(),
                    time_points_seek_bo_df.median(),
                    time_points_hide_df.median())


def make_behavioral_states_str(trial, playing_states=True, extra_states=True, observing_states=True, combine_observing_states=False):
    """
    Make a DataArray containing the behavioral state at every time point (using the time grid of trial.states) in the trial as a string
    Assumes states have been added to the trial

    Parameters
    ----------
    trial : Trial
        trial object
    playing_states : bool, default True
        include playing states which are phases of the game
        and potentially darting and exploring (see extra_states)
    extra_states : bool, default True
        if including playing states, also include darting and exploring
    observing_states : bool, default True
        include observing states (engaged, grooming, resting observing)
        NOTE: if playing_states is False, outside tagged intervals the state will be called 'random_observing', so only use this combination with observing trials
    combine_observing_states : bool, default False
        instead of engaged, grooming, resting use combine these and use
        just one 'observing' state
    
    Returns
    -------
    st : xr.DataArray
        DataArray containing the behavioral states
    """
    try:
        st = trial.states.copy()
    except:
        st = trial.factors.isel(factor = 0).copy()

    st[:] = -1
    st = st.astype(str)

    if playing_states:
        if trial.role == 'seek':
            st.loc[{'time' : slice(trial.time_points.start, trial.time_points.box_open)}] = 'box_start_closed'
            st.loc[{'time' : slice(trial.time_points.box_open, trial.time_points.jump_out)}] = 'box_start_open'
            st.loc[{'time' : slice(trial.time_points.jump_out, trial.time_points.interaction)}] = 'game_seek'
            st.loc[{'time' : slice(trial.time_points.interaction, trial.time_points.transit)}] = 'interaction'
            st.loc[{'time' : slice(trial.time_points.transit, trial.time_points.jump_in)}] = 'transit'
            st.loc[{'time' : slice(trial.time_points.jump_in, trial.time_points.end)}] = 'box_end_open'

        if trial.role == 'hide':
            st.loc[{'time' : slice(trial.time_points.start, trial.time_points.jump_out)}] = 'box_start_open'
            st.loc[{'time' : slice(trial.time_points.jump_out, trial.time_points.interaction)}] = 'game_hide'
            st.loc[{'time' : slice(trial.time_points.interaction, trial.time_points.transit)}] = 'interaction'
            st.loc[{'time' : slice(trial.time_points.transit, trial.time_points.jump_in)}] = 'transit'
            st.loc[{'time' : slice(trial.time_points.jump_in, trial.time_points.end)}] = 'box_end_open'

        if extra_states:
            rename_state_intervals(st, trial.darting_start_times, trial.darting_end_times, 'darting')
            rename_state_intervals(st, trial.exploring_start_times, trial.exploring_end_times, 'exploring')
            rename_state_intervals(st, trial.hiding_start_times, trial.hiding_end_times, 'hiding')

    if observing_states:
        if combine_observing_states:
            rename_state_intervals(st, trial.observing_start_times, trial.observing_end_times, 'observing')
        else: 
            rename_state_intervals(st, trial.engaged_observ_start_times, trial.engaged_observ_end_times, 'engaged_observing') 
            rename_state_intervals(st, trial.grooming_observ_start_times, trial.grooming_observ_end_times, 'grooming_observing') 
            rename_state_intervals(st, trial.resting_observ_start_times, trial.resting_observ_end_times, 'resting_observing')
            
        if not playing_states:
            st[st == '-1'] = 'random_observing'
                
    return st 


def rename_state_intervals(st, start_times, end_times, state_name):
    """
    Rename states in array between start_times and end_times to state_name
    with correcting for the cases when start_times and end_times do not define
    proper intervals (e.g. one of them has more elements or end times are before start times)
    Attention: modifies st in place!

    Parameters
    ----------
    st : xr.DataArray
        array containing the current states
        Attention: gets modified!
    start_times : array-like
        start times of the intervals
    end_times : array-like
        end times of the intervals
    state_name : str or int (dtype of st)
        value for the states in the intervals

    Returns
    -------
    None, but modifies st
    """
    starts, ends = correct_starts_and_ends(start_times, end_times)
    
    for start, end in zip(starts, ends):
        assert (end is None) or (start < end), (start_times, end_times)
        st.loc[{'time' : slice(start, end)}] = state_name


def convert_str_states_to_int(states, state_dict):
    """
    Convert an array containing states given as strings to an array containing states as integers
    based on a mapping from strings to integers

    Parameters
    ----------
    states : xr.DataArray
        states as strings
    state_dict : dict or None
        mapping from str to int
        if None: map states to range(number of unique states)

    Returns
    -------
    st : xr.DataArray
        mapped DataArray with dtype int
    """
    st = states.copy()
    
    if state_dict is None:
        state_dict = {name : num for num, name in enumerate(np.unique(st))}
        
    for name in np.unique(st):
        st[st == name] = state_dict[name]
        
    return st.astype(int)


def make_behavioral_states_int(trial, state_dict, playing_states=True, extra_states=True, observing_states=True, combine_observing_states=False):
    """
    Make a DataArray containing the behavioral state at every time point (using the time grid of trial.states) in the trial as an integer
    Assumes states have been added to the trial

    Parameters
    ----------
    trial : Trial
        trial object
    state_dict : dict or None
        mapping from str to int
        if None: map states to range(number of unique states)
    playing_states : bool, default True
        include playing states which are phases of the game
        and potentially darting and exploring (see extra_states)
    extra_states : bool, default True
        if including playing states, also include darting and exploring
    observing_states : bool, default True
        include observing states (engaged, grooming, resting observing)
        NOTE: if playing_states is False, outside tagged intervals the state will be called 'random_observing', so only use this combination with observing trials
    combine_observing_states : bool, default False
        instead of engaged, grooming, resting use combine these and use
        just one 'observing' state

    Returns
    -------
    st : xr.DataArray
        DataArray containing the behavioral states
    """
    states = make_behavioral_states_str(trial,
                                        playing_states = playing_states,
                                        extra_states = extra_states,
                                        observing_states = observing_states,
                                        combine_observing_states = combine_observing_states)
    
    return convert_str_states_to_int(states, state_dict)


def contingency_xr(behavioral_states, hmm_states, return_probs=False):
    """
    Create a contingency matrix for the behavioral states and HMM states with coordinates
    
    Parameters
    ----------
    behavioral_states : 1D array-like
        behavioral states
    hmm_states : 1D array-like
        HMM states
    return_probs : bool, default False
        normalize the counts by the length of the behavioral states
        gives ~ probability of the hmm_state in the behavioral state
        
    Returns
    -------
    cont_mx : xr.DataArray
        (behavioral_state x hmm_state) contingency matrix with coordinates
    """
    if isinstance(behavioral_states, xr.DataArray):
        behavioral_states = behavioral_states.values
    if isinstance(hmm_states, xr.DataArray):
        hmm_states = hmm_states.values

    cont_mx = xr.DataArray(sklearn.metrics.cluster.contingency_matrix(behavioral_states, hmm_states),
                           dims = ('behavioral_state', 'hmm_state'))
    cont_mx = cont_mx.assign_coords({'behavioral_state' : np.unique(behavioral_states),
                                     'hmm_state' : np.unique(hmm_states)})
    
    if return_probs:
        behavioral_state_counts = pd.Series({id : count
                                             for (id, count)
                                             in zip(*np.unique(behavioral_states,
                                                               return_counts = True))}).rename_axis('behavioral_state').to_xarray()
        return cont_mx / behavioral_state_counts
    
    return cont_mx


def relabel_partitions(states):
    """
    Change the states to integers switching at every transition point to keep transition points but lose state identities
    (every partition has its own ID)
    
    Parameters
    ----------
    states : xr.DataArray
        states from where to get the transition points
        
    Returns
    -------
    st : xr.DataArray
        DataArray with the same transition points but meaningless state IDs
    """
    st = states.copy()
    
    # convert to integer to be able to calculate the transition points
    # TODO: might want to put this into get_transition_indices and get_transition_points
    if not np.issubdtype(st.dtype, np.number):
        st = convert_str_states_to_int(st, None)
        
    transition_indices = get_transition_indices(st)
    
    # if there are no transitions, the whole thing has just one state
    if len(transition_indices) == 0:
        st[:] = 0
        return st.astype(int)
        
    left_borders = np.insert(transition_indices, 0, 0)
    
    for i, (start, end) in enumerate(zip(left_borders[:-1], left_borders[1:])):
        st[start:end] = i


    st[left_borders[-1]:] = i + 1
    
    return st


def randomize_state_labels(states, K, N=1):
    """
    Keep the transition points, but assign a random state ID to every segment
    
    Parameters
    ----------
    states : xr.DataArray with a time dimension
        original discrete states
    K : int or str
        if int: number of possible states
        if 'same': use the number of distrinct states in states
    N : int, default 1
        number of random arrays to create
        
    Returns
    -------
    st : xr.DataArray
        states with same transition points as the original, but
        with new values where every segment has a random state value
    """
    st = states.copy()
    
    # convert to integer to be able to calculate the transition points
    # TODO: might want to put this into get_transition_indices and get_transition_points
    if not np.issubdtype(st.dtype, np.number):
        st = convert_str_states_to_int(st, None)
        
    transition_indices = get_transition_indices(st)
    
    if K == 'same':
        K = len(np.unique(st.values))
    elif not isinstance(K, int):
        raise ValueError("K has to be an integer of 'same'")
    
    def make_one_array():
        # if there are no transitions, the whole thing has just one state
        if len(transition_indices) == 0:
            new_arr = np.zeros_like(st.values)
        else:
            new_arr = np.empty_like(st.values)
            left_borders = np.insert(transition_indices, 0, 0)

            for start, end in zip(left_borders[:-1], left_borders[1:]):
                assert start < end
                new_arr[start:end] = np.random.randint(K)

            new_arr[left_borders[-1]:] = np.random.randint(K)
            
        return xr.DataArray(new_arr, dims = ['time'], coords = {'time' : st.time.values})
    
    if N == 1:
        return make_one_array()
    else:
        return [make_one_array() for i in range(N)]
