import numpy as np
import pandas as pd
import xarray as xr

from .misc import *
from .stretching import stretch_states, stretch_2d

def get_states_around_time_points_in_session(session, point_name, before_length, after_length, role, return_xr=True, dim_name=None):
    """
    States around the specified tagged points

    Parameters
    ----------
    session : Session
        session to extract for
    point_name : str
        name of the attribute which stores the time points around which we want to get the states 
    before_length : int
        how much before the point start extracting the states
    after_length : int
        how much after the point start extracting the states
    role : str
        the role in which to extract the states
        seek, hide or both
    return_xr : bool, default True
        concatenate the list of states into an xr.DataArray
        and return that
    dim_name : str, optional
        name of the concatenation dimension of the DataArray
        if return_xr is True

    Returns
    -------
    list of length # of occurrences of the given type of time point
    in the session containing xr.DataArrays with the states
    around the time points
    or that list concatenated if return_xr is True
    """
    extr_states = []
    for i, trial in enumerate(session.trials):
        if role not in [trial.role, 'both']:
            continue

        states = trial.states
        for ds in getattr(trial, point_name):
            # might have to multiply the state.time.values by bin_length
            extr_states.append(states.sel(time = slice(ds - before_length, ds + after_length)))

    if return_xr:
        if dim_name is None:
            dim_name = point_name
        return concat_extracted_states(extr_states, dim_name)
    else:
        return extr_states


def get_state_probs_around_time_points_in_session(session, point_name, before_length, after_length, role, dim_name=None):
    """
    States probabilities around the specified tagged points

    Parameters
    ----------
    session : Session
        session to extract for
    point_name : str
        name of the attribute which stores the time points around which we want to get the states 
    before_length : int
        how much before the point start extracting the states
    after_length : int
        how much after the point start extracting the states
    role : str
        the role in which to extract the states
        seek, hide or both
    dim_name : str, optional
        name of the concatenation dimension of the DataArray
        if return_xr is True

    Returns
    -------
    xr.DataArray with dimensions: dim x state x time
    """
    extr_states = []
    for i, trial in enumerate(session.trials):
        if role not in [trial.role, 'both']:
            continue

        for ds in getattr(trial, point_name):
            extr_states.append(trial.state_probs.sel(time = slice(ds - before_length, ds + after_length))) 

    if dim_name is None:
        dim_name = point_name
    return concat_extracted_states(extr_states, dim_name)


def get_states_between_time_points_in_trial(trial, start_point_name, end_point_name, start_pad=0., end_pad=0., return_xr=True, dim_name=None):
    """
    States between two specified tagged points

    Parameters
    ----------
    trial : Trial
        trial to extract for
    start_point_name : str
        name of the attribute which stores the time points
        that serve as starting points for the state extraction
    end_point_name : str
        name of the attribute which stores the time points
        that serve as end points for the state extraction
    start_pad : float
        extract states from start_pad ms before the start point
    end_pad : float
        extract states until end_pad ms after the end point
    return_xr : bool, default True
        concatenate the list of states into an xr.DataArray
        and return that
    dim_name : str, optional
        name of the concatenation dimension of the DataArray
        if return_xr is True

    Returns
    -------
    list of length # of occurrences of the given type of time point
    in the session containing xr.DataArrays with the states
    between the time points
    or that list concatenated if return_xr is True
    """
    extr_states = []

    starts = getattr(trial, start_point_name)
    ends = getattr(trial, end_point_name)

    starts, ends = correct_starts_and_ends(starts, ends)

    for start, end in zip(starts, ends):
        if end is None:
            end = float(trial.states.time[-1])
        binning = float(trial.states.time[1] - trial.states.time[0])
        extr_states.append(trial.states.sel(time = slice(start - start_pad - binning/2, end + end_pad + binning/2)))
    
    if return_xr:
        if dim_name is None:
            dim_name = start_point_name
        return concat_extracted_states(extr_states, dim_name)
    else:
        return extr_states


def get_states_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad=0., end_pad=0., return_xr=True, dim_name=None):
    """
    States between two specified tagged points

    Parameters
    ----------
    session : Session
        session to extract for
    start_point_name : str
        name of the attribute which stores the time points
        that serve as starting points for the state extraction
    end_point_name : str
        name of the attribute which stores the time points
        that serve as end points for the state extraction
    role : str
        the role in which to extract the states
        seek, hide or both
    start_pad : float
        extract states from start_pad ms before the start point
    end_pad : float
        extract states until end_pad ms after the end point
    return_xr : bool, default True
        concatenate the list of states into an xr.DataArray
        and return that
    dim_name : str, optional
        name of the concatenation dimension of the DataArray
        if return_xr is True

    Returns
    -------
    list of length # of occurrences of the given type of time point
    in the session containing xr.DataArrays with the states
    between the time points
    or that list concatenated if return_xr is True
    """
    extr_states = []
    for trial in session.trials:
        if role not in [trial.role, 'both']:
            continue
        
        extr_states = extr_states + get_states_between_time_points_in_trial(trial, start_point_name, end_point_name, start_pad, end_pad, return_xr=False)

    if return_xr:
        if dim_name is None:
            dim_name = start_point_name
        return concat_extracted_states(extr_states, dim_name)
    else:
        return extr_states


def get_state_probs_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad=0., end_pad=0., return_xr=True, dim_name=None):
    """
    State probabilities between two specified tagged points

    Parameters
    ----------
    session : session
        session to extract for
    start_point_name : str
        name of the attribute which stores the time points
        that serve as starting points for the state extraction
    end_point_name : str
        name of the attribute which stores the time points
        that serve as end points for the state extraction
    role : str
        the role in which to extract the states probabilities
        seek, hide or both
    start_pad : float
        extract states probabilities from start_pad ms before the start point
    end_pad : float
        extract states probabilities until end_pad ms after the end point
    return_xr : bool, default true
        concatenate the list of probabilities into an xr.dataarray
        and return that
    dim_name : str, optional
        name of the concatenation dimension of the dataarray
        if return_xr is true

    returns
    -------
    xr.dataarray with dimensions: dim x state x time
    """
    extr_probs = []
    for trial in session.trials:
        if role not in [trial.role, 'both']:
            continue
        
        starts = getattr(trial, start_point_name)
        ends = getattr(trial, end_point_name)
        starts, ends = correct_starts_and_ends(starts, ends)

        for start, end in zip(starts, ends):
            if end is None:
                end = float(trial.states.time[-1])
            #extr_probs.append(trial.state_probs.sel(time = slice(start, end))) 
            binning = float(trial.state_probs.time[1] - trial.state_probs.time[0])
            extr_probs.append(trial.state_probs.sel(time = slice(start - start_pad - binning/2, end + end_pad + binning/2)))

    if return_xr:
        if dim_name is None:
            dim_name = point_name
        return concat_extracted_states(extr_probs, dim_name)
    else:
        return extr_probs


def get_stretched_state_probs_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad=0., end_pad=0., dim_name=None):
    """
    State probabilities between two specified tagged points in a session
    stretched to the same length

    parameters
    ----------
    session : session
        session to extract for
    start_point_name : str
        name of the attribute which stores the time points
        that serve as starting points for the state extraction
    end_point_name : str
        name of the attribute which stores the time points
        that serve as end points for the state extraction
    role : str
        the role in which to extract the states probabilities
        seek, hide or both
    start_pad : float
        extract states probabilities from start_pad ms before the start point
    end_pad : float
        extract states probabilities until end_pad ms after the end point
    dim_name : str, optional
        name of the concatenation dimension of the dataarray
        if return_xr is true

    Returns
    -------
    xr.dataarray with dimensions: dim x state x time
    """

    probs_list = get_state_probs_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad, end_pad, dim_name = dim_name, return_xr=False)
    
    for x in probs_list:
        x['time'] = x.time - x.time[0]
        
    median_end = np.median([x.time[-1] for x in probs_list])
    binning = probs_list[0].time.values[1] - probs_list[0].time.values[0]
    
    stretched_list = [stretch_2d(x,
                                 [0., start_pad, x.time.values[-1] - end_pad, x.time.values[-1]],
                                 [0., start_pad, median_end - end_pad, median_end],
                                 binning, 'numpy', True)
                      for x in probs_list]
    
    if dim_name is None:
        dim_name = start_point_name
    return xr.concat(stretched_list, dim = dim_name)


def get_stretched_states_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad=0., end_pad=0., dim_name=None):
    """
    States between two specified tagged points in a session
    stretched to the same length

    parameters
    ----------
    session : session
        session to extract for
    start_point_name : str
        name of the attribute which stores the time points
        that serve as starting points for the state extraction
    end_point_name : str
        name of the attribute which stores the time points
        that serve as end points for the state extraction
    role : str
        the role in which to extract the states probabilities
        seek, hide or both
    start_pad : float
        extract states probabilities from start_pad ms before the start point
    end_pad : float
        extract states probabilities until end_pad ms after the end point
    dim_name : str, optional
        name of the concatenation dimension of the dataarray
        if return_xr is true

    returns
    -------
    xr.dataarray with dimensions: dim x state x time
    """

    states_list = get_states_between_time_points_in_session(session, start_point_name, end_point_name, role, start_pad, end_pad, dim_name = dim_name, return_xr=False)
    
    states_list = [x for x in states_list if x.time.size > 0]
    for x in states_list:
        x['time'] = x.time - x.time[0]
        
    median_end = np.median([x.time[-1] for x in states_list])
    binning = states_list[0].time.values[1] - states_list[0].time.values[0]
    
    stretched_list = [stretch_states(x,
                                     [0., start_pad, x.time.values[-1] - end_pad, x.time.values[-1]],
                                     [0., start_pad, median_end - end_pad, median_end],
                                     binning, True)
                      for x in states_list]
    
    if dim_name is None:
        dim_name = start_point_name
    return xr.concat(stretched_list, dim = dim_name)


def concat_extracted_states(states, dim):
    """
    Concatenate a list of states extracted around time points by get_states_around_time_points_in_session
    
    Parameters
    ----------
    states : list of xr.DataArrays
        list of states extracted around each of the time points
    dim : str
        new dimension name to concatenate along

    Returns
    -------
    session_states : xr.DataArray
        states concatenated by reindexing the time dimension, then alinging on time
        and concatenating along the given new dimension
    """
    with_range_time = [x.assign_coords({'time' : range(len(x.time))}) for x in states]
    return xr.concat(with_range_time, compat = 'no_conflicts', dim = dim)
