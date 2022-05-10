import numpy as np
import xarray as xr

import scipy.signal as scs
from scipy.interpolate import interp1d

def filter_between(array, start, end, inclusive=True):
    """
    Filter an array between two values

    Parameters
    ----------
    array : array-like
        array whose values to filter
    start : float
        lower bound on values in the resulting array
    end : float
        upper bound on values in the resulting array
    inclusive : bool, default True
        whether to include lower and upper bound

    Returns
    -------
    an array with elements of the original array that are between
    start and end
    """
    if isinstance(array, (list, tuple)):
        farray = np.array(array)
    else:
        farray = array

    if inclusive:
        return farray[np.logical_and(start <= farray, farray <= end)]
    else:
        return farray[np.logical_and(start < farray, farray < end)]


def norm_gauss_window(bin_length, std):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_length : float
        binning length of the array we want to smooth in ms
    std : float
        standard deviation of the window
        use hw_to_std to calculate std based from half-width
    """
    win = scs.gaussian(int(5*std/bin_length), std/bin_length)
    return win / np.sum(win)


def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))


def create_spike_train(spike_times, tend, bin_length):
    """
    Create spike train from spike times by binning

    Parameters
    ----------
    spike_times : array-like
        spike times to smooth
    tend : float
        length of time
    bin_length : float
        spike binning window length in ms

    Returns
    -------
    xr.DataArray of the rates with time coordinates
    """
    time = np.arange(0, tend, bin_length)
    spike_train, bins = np.histogram(spike_times, np.append(time, tend))

    return xr.DataArray(spike_train, dims=('time'), coords={'time' : time})


def create_smoothed_rate(spike_times, tend, bin_length, smoothing_win=None, smoothing_hw=None):
    """
    Create smoothed rate (in Hz) from spike times by first binning, then convolving by the given smoothing window

    Parameters
    ----------
    spike_times : array-like
        spike times to smooth
    tend : float
        length of time
    bin_length : float
        spike binning window length in ms
    smoothing_win : array-like, default None
        smoothing window
        NOTE: normalize mass to get results in Hz
        e.g. see preproc.norm_gauss_window
        provide either this or smoothing_hw
    smoothing_hw : float, default None
        convolve with a normalized Gaussian window with this half width in ms
        provide either this or smoothing_win

    Returns
    -------
    xr.DataArray of the rates in Hz with time coordinates
    """
    if (smoothing_hw is not None) and (smoothing_win is not None):
        raise Exception('only provide smoothing_hw or smoothing_win')
    if (smoothing_win is None) and (smoothing_hw is None):
        raise Exception('provide either smoothing_win or smoothing_hw')
    
    # if smoothing_hw is provided, create smoothing window with the appropriate half-width
    if smoothing_hw is not None:
        smoothing_win = norm_gauss_window(bin_length, hw_to_std(smoothing_hw))

    # could add option to specify padding
    spike_train = create_spike_train(spike_times, tend, bin_length)
    rate = scs.convolve(spike_train.values * (1000 / bin_length), smoothing_win, mode='same')

    return xr.DataArray(rate, dims=('time'), coords={'time' : spike_train.time})


def smooth_spike_train(spike_train, bin_length=None, smoothing_win=None, smoothing_hw=None):
    """
    Smooth a spike train to make firing rates

    Parameters
    ----------
    spike_train : xr.DataArray
        spike train to smooth
        can be 1 or 2D
    bin_length : float
        spike binning window length in ms
        if None, it's inferred from the spike train
    smoothing_win : array-like, default None
        smoothing window
        NOTE: normalize mass to get results in Hz
        e.g. see preproc.norm_gauss_window
        provide either this or smoothing_hw
    smoothing_hw : float, default None
        convolve with a normalized Gaussian window with this half width in ms
        provide either this or smoothing_win

    Returns
    -------
    xr.DataArray of the rates in Hz with time coordinates
    """
    if bin_length is None:
        diffs = set(np.diff(spike_train.time))
        assert len(diffs) == 1
        bin_length = list(diffs)[0]

    if (smoothing_hw is not None) and (smoothing_win is not None):
        raise Exception('only provide smoothing_hw or smoothing_win')
    if (smoothing_win is None) and (smoothing_hw is None):
        raise Exception('provide either smoothing_win or smoothing_hw')
    
    # if smoothing_hw is provided, create smoothing window with the appropriate half-width
    if smoothing_hw is not None:
        smoothing_win = norm_gauss_window(bin_length, hw_to_std(smoothing_hw))

    def sm(neur_train):
        rate = scs.convolve(neur_train.values * (1000 / bin_length), smoothing_win, mode='same')
        return xr.DataArray(rate, dims=('time'), coords={'time' : neur_train.time})

    if spike_train.ndim > 2:
        raise NotImplementedError('smooth_spike_train takes 1 or 2 dimensional arrays')
    elif spike_train.ndim == 2:
        # the non-time dimension
        other_dim = [dim_name for dim_name in spike_train.dims if dim_name != 'time'][0]
        rates = xr.concat([sm(spike_train.sel({other_dim : i})) for i in spike_train[other_dim]],
                          dim = other_dim)
        return rates.assign_coords({other_dim : spike_train[other_dim]})
    else:
        return sm(spike_train)


def z_score_xr(dr):
    """
    Remove the average firing rate and divide by the standard deviation for every neuron.
    """
    other_dims = [d for d in dr.dims if d != 'neuron']
    return (dr - dr.mean(other_dims)) / dr.std(other_dims)


def scale_xr(da):
    """
    Set the minimum across time to zero, then divide by the maximum across time to get a signal
    that is between 0 and 1

    Parameters
    ----------
    da : xr.DataArray
        data with a time dimension
        e.g. neuron x time

    Returns
    -------
    scaled DataArray
    """
    scaled = da - da.min('time')
    return scaled / scaled.max('time')


default_bin_length = 5
default_half_width = 250
default_smoothing_window = norm_gauss_window(default_bin_length, hw_to_std(default_half_width))

