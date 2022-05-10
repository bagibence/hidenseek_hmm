import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import interp1d

def stretch_array(arr, old_anchors, new_anchors, dt, interp='numpy', shift_to_zero=True):
    """
    Stretch time in an array to align with different arrays
    
    Parameters
    ----------
    arr : xr.DataArray
        array with time coordinates
    old_anchors : array-like
        list of original time points
        has to include start and end
    new_anchors : array-like
        list of new time points to align on
        has to include start and end
    dt : float
        sampling length in ms
    interp : str, default 'numpy'
        interpolation to use
        use 'nearest' for discrete states
        use 'numpy' or 'linear' for continuous values (e.g. smooth firing rates)
    shift_to_zero : bool, default True
        shift time start to zero before alignment
        use True if the anchor points are relative to the array start
        
    Returns
    -------
    xr.DataArray with time stretched to align old_anchors to new_anchors
    """
    
    if shift_to_zero:
        arr['time'] = arr.time - arr.time[0]
    vals = arr.values
    tt = arr.time.values
        
    time = np.arange(new_anchors[0], new_anchors[-1], dt)
    stretched = np.zeros(len(time))
    
    try:
        keep_ind = ~np.isnan(old_anchors)
        old_anchors = np.array(old_anchors)[keep_ind]
        new_anchors = np.array(new_anchors)[keep_ind]
    except:
        print(old_anchors)
        print(new_anchors)

    for k in range(len(new_anchors)-1):
        indtofill = np.argwhere((new_anchors[k] <= time) & (time <= new_anchors[k+1]))
        compress_factor = (old_anchors[k+1] - old_anchors[k]) / (new_anchors[k+1] - new_anchors[k])
        timeint = old_anchors[k] + compress_factor * (time[indtofill] - new_anchors[k])

        if interp == 'numpy':
            stretched[indtofill] = np.interp(timeint, tt, vals)
        elif interp in ['nearest', 'linear']:
            stretched[indtofill] = interp1d(tt, vals, kind=interp, bounds_error=False, fill_value='extrapolate')(timeint)
        else:
            raise Exception(f'{interp} is not a valid interpolation method')


    stretched[-1] = vals[-1]

    return xr.DataArray(stretched, dims=('time'), coords={'time' : time})


def stretch_2d(arr, old_anchors, new_anchors, dt, interp='numpy', shift_to_zero=True):
    """
    Stretch time in a 2D array to align with different arrays
    
    Parameters
    ----------
    arr : xr.DataArray
        array with something x time coordinates
    old_anchors : array-like
        list of original time points
        has to include start and end
    new_anchors : array-like
        list of new time points to align on
        has to include start and end
    dt : float
        sampling length in ms
    interp : str, default 'numpy'
        interpolation to use
        use 'nearest' for discrete states
        use 'numpy' or 'linear' for continuous values (e.g. smooth firing rates)
    shift_to_zero : bool, default True
        shift time start to zero before alignment
        use True if the anchor points are relative to the array start
        
    Returns
    -------
    xr.DataArray with time stretched to align old_anchors to new_anchors
    """
    non_time_dim = [d for d in arr.dims if d != 'time']
    assert len(non_time_dim) == 1, 'DataArray can only have one non-time dimension'
    non_time_dim = non_time_dim[0]
    
    return xr.concat([stretch_array(arr.sel({non_time_dim : i}), old_anchors, new_anchors, dt, interp, shift_to_zero)
                      for i in arr[non_time_dim]],
                     dim = non_time_dim)


def stretch_states(arr, old_anchors, new_anchors, dt, shift_to_zero=True):
    """
    Stretch time in an array (1D or 2D) containing discrete states to align with different arrays
    
    Parameters
    ----------
    arr : xr.DataArray
        array with time coordinates
    old_anchors : array-like
        list of original time points
        has to include start and end
    new_anchors : array-like
        list of new time points to align on
        has to include start and end
    dt : float
        sampling length in ms
    shift_to_zero : bool, default True
        shift time start to zero before alignment
        use True if the anchor points are relative to the array start
        
    Returns
    -------
    xr.DataArray with time stretched to align old_anchors to new_anchors
    """
    if len(arr.dims) == 1:
        assert arr.dims[0] == 'time', 'DataArray needs to have a time dimension'
        return stretch_array(arr, old_anchors, new_anchors, dt, 'nearest', shift_to_zero)
    elif len(arr.dims) == 2:
        return stretch_2d(arr, old_anchors, new_anchors, dt, 'nearest', shift_to_zero)
    else:
        raise Exception('check your DataArray')


def stretch_rates(arr, old_anchors, new_anchors, dt, shift_to_zero=True):
    """
    Stretch time in an array (1D or 2D) containing smooth rates to align with different arrays
    
    Parameters
    ----------
    arr : xr.DataArray
        array with time coordinates
    old_anchors : array-like
        list of original time points
        has to include start and end
    new_anchors : array-like
        list of new time points to align on
        has to include start and end
    dt : float
        sampling length in ms
    shift_to_zero : bool, default True
        shift time start to zero before alignment
        use True if the anchor points are relative to the array start
        
    Returns
    -------
    xr.DataArray with time stretched to align old_anchors to new_anchors
    """
    if len(arr.dims) == 1:
        assert arr.dims[0] == 'time', 'DataArray needs to have a time dimension'
        return stretch_array(arr, old_anchors, new_anchors, dt, 'numpy', shift_to_zero)
    elif len(arr.dims) == 2:
        return stretch_2d(arr, old_anchors, new_anchors, dt, 'numpy', shift_to_zero)
    else:
        raise Exception('check your DataArray')


