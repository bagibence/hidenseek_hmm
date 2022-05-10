import numpy as np
import pandas as pd
import xarray as xr


def get_transition_points(states):
    """
    Return the times when there is a state transition occurring in the states
    (first time coordinate where the new state is active) 

    Parameters
    ----------
    states : xr.DataArray
        inferred states for a trial

    Returns
    -------
    transition_points : np.array
    """
    return np.array(states.time[1:][(states.diff('time') != 0)])


def get_transition_indices(states):
    """
    Get the indices of the (right side) of transition points in an array of states
    
    Parameters
    ----------
    states : xr.DataArray with a time coordinate or 1D np.array
        array with the discrete states
    
    Returns
    -------
    np.array with the indices
    """
    index = np.arange(len(states))

    if isinstance(states, xr.DataArray):
        return np.array(index[1:][(states.diff('time') != 0)])
    elif isinstance(states, np.ndarray) and states.ndim == 1:
        return np.array(index[1:][(np.diff(states) != 0)])
    else:
        raise ValueError('states has to be xr.DataArray with a time dimension or a 1D np.ndarray')


def get_transition_state_pairs(states):
    """
    Return from which to which state there was a transition in the trial

    Parameters
    ----------
    states : xr.DataArray
        inferred states for a trial

    Returns
    -------
    list of tuples like (from_state, to_state)
    """
    from_indices = np.where(states.diff('time') != 0)[0]
    to_indices = from_indices + 1

    from_states = np.array(states[from_indices])
    to_states = np.array(states[to_indices])

    return list(zip(from_states, to_states))


def sample_random_transition_points(states, N, seed, exclude_borders=True):
    """
    Sample random transition points from the time grid of an array of states
    
    Parameters
    ----------
    states : xr.DataArray
        states (contatining discrete states) for which to sample
        (only its time coordinate matters)
    N : int
        number of times to repeat
        (e.g. when generating surrogate data for significance testing)
    seed : int
        random seed
    exclude_borders : bool, default True
        first and last time point cannot be sampled if True
        
    Returns
    -------
    points : np.array or list of np.arrays
        time points at which fake transition points were sampled
        same length as the number of real transition points
        one array if N is 1
        list of such arrays if N > 1
    """
    np.random.seed(seed)
    assert N >= 1, 'number of sampled arrays has to be at least 1'
    
    n_points = len(get_transition_points(states))
    
    if exclude_borders:
        time_grid = states.time.values[1:-1]
    else:
        time_grid = states.time.values
        
    if N == 1:
        return np.sort(np.random.choice(time_grid, n_points, replace = False))
    else:
        return [np.sort(np.random.choice(time_grid, n_points, replace = False)) for i in range(N)]


def sample_random_transition_indices(states, N, seed, exclude_borders=True):
    """
    Sample random transition indices from the time grid of an array of states
    
    Parameters
    ----------
    states : xr.DataArray
        states (contatining discrete states) for which to sample
        (only its time coordinate matters)
    N : int
        number of times to repeat
        (e.g. when generating surrogate data for significance testing)
    seed : int
        random seed
    exclude_borders : bool, default True
        first and last time point cannot be sampled if True
        
    Returns
    -------
    points : np.array or list of np.arrays
        time points at which fake transition points were sampled
        same length as the number of real transition points
        one array if N is 1
        list of such arrays if N > 1
    """
    np.random.seed(seed)
    assert N >= 1, 'number of sampled arrays has to be at least 1'
    
    n_points = len(get_transition_points(states))
    
    time_indices = np.arange(len(states.time))
    if exclude_borders:
        time_grid = time_indices[1:-1]
    else:
        time_grid = time_indices
        
    if N == 1:
        return np.sort(np.random.choice(time_grid, n_points, replace = False))
    else:
        return [np.sort(np.random.choice(time_grid, n_points, replace = False)) for i in range(N)]


def build_states_from_transition_indices(time_grid, transition_indices):
    """
    Build an array of discrete states with a different state introduced at every transition point
    
    Parameters
    ----------
    time_grid : np.array
        time coordinates on which to build the array with the new states
    transition_indices : np.array
        indices of time_grid at which to introduce a new state
        
    Returns
    -------
    xr.DataArray (int dtype) with time coordinates time_grid and
    a new state introduced at every point in transition_indices
    """
    if len(transition_indices) == 0:
        new_arr = np.zeros_like(time_grid)
    else:
        left_borders = np.insert(np.sort(transition_indices), 0, 0)
        new_arr = np.empty_like(time_grid)
        for i, (start, end) in enumerate(zip(left_borders[:-1], left_borders[1:])):
            new_arr[start:end] = i
        new_arr[left_borders[-1]:] = i + 1
    
    return xr.DataArray(new_arr, dims = ['time'], coords = {'time' : time_grid}).astype(int)


def generate_fake_states_with_same_number_of_transitions(real_states, N, seed):
    """
    Generate fake states with the same time coordinates and same number of transition points as real_states
    
    Parameters
    ----------
    real_states : xr.DataArray
        real states whose time coordinates and number of transitions to mimic
    N : int
        number of fake state arrays to generate
    seed : int
        random seed
        
    Returns
    -------
    fake_states_list : (list of) xr.DataArray(s)
        if N is 1, a DataArray
        if N > 1, a list of DataArrays
    """
    random_transition_indices_list = sample_random_transition_indices(real_states, N, seed)
    if N == 1:
        # it's not actually a list, just one np.array
        return build_states_from_transition_indices(real_states.time, random_transition_indices_list)
        
    fake_states_list = [build_states_from_transition_indices(real_states.time, fake_indices)
                        for fake_indices in random_transition_indices_list]
    
    return fake_states_list
