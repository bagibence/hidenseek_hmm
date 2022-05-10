import xarray as xr
import numpy as np


def split_array(X, lengths):
    """
    Split an array into subarrays of given lengths along its first dimension.

    Parameters
    ----------
    X : np.array
        array to split
    lengths : list of int
        lengths of the segments

    Returns
    -------
    list of subarrays
    """
    assert np.sum(lengths) == X.shape[0]
    
    border_indices = np.insert(np.cumsum(lengths), 0, 0)
    return [X[start:end, ...] for (start, end) in zip(border_indices[:-1], border_indices[1:])]


def reduce_dim(xarr, f, dim):
    """
    Apply function f to xr.DataArray reducing dimension dim

    Parameters
    ----------
    xarr : xr.DataArray
        dataarray to process
    f : function
        function to apply
    dim : string
        name of the dimension to reduce

    Returns
    -------
    reduced xarr (without dim)
    """
    other_dims = tuple(d for d in xarr.dims if d != dim)
    return xr.apply_ufunc(f,
                          xarr,
                          input_core_dims = ([dim, *other_dims], ),
                          exclude_dims = {dim, },
                          output_core_dims = (other_dims, ),
                          kwargs = {'axis' : 0})

def _apply(self, f, dim):
    return reduce_dim(self, f, dim)
xr.DataArray.apply = _apply

