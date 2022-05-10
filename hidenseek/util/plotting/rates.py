import xarray as xr

import seaborn as sns

import warnings


def rates_to_df(arr):
    """
    Convert DataArray of rates to DataFrame

    Parameters
    ----------
    arr : xr.DataArray
        DataArray containing the rates of neurons across time

    Returns
    -------
    pd.DataFrame with neurons as index and time points as columns
    """
    non_time_dims = [dim for dim in arr.dims if dim != 'time']
    assert len(non_time_dims) == 1
    non_time_dim = non_time_dims[0]

    return (arr.to_dataframe('name')
               .reset_index()
               .pivot(index = non_time_dim, columns = 'time', values = 'name'))


def plot_rasterlike_rates(ax, rates, normalize=True, cbar_ax=None, cmap=None):
    """
    Plot rates of a trial similarly to a rasterplot

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    rates : pd.DataFrame
        rates to plot
        neuron x time
        convert DataArray to DataFrame by rates_to_df
    normalize : bool (default True)
        normalize every neuron's max. firing rate to 1
    cbar_ax : plt.Axes (default None)
        Axes to plot the colorbar on
        if None, don't plot a colorbar
    cmap : string or colormap
        colormap to use for the rates
    """
    if normalize:
        rates /= rates.max('time')

    if cbar_ax is not None:
        cbar = True
    else:
        cbar = False

    if cmap is None:
        cmap = sns.color_palette('Wistia', desat = 0.6)

    if isinstance(rates, xr.DataArray):
        plot_rates = rates_to_df(rates)
    else:
        plot_rates = rates
        warning.warn('would be nice to get an indexed DataArray')

    sns.heatmap(plot_rates, ax = ax, cbar = cbar, cbar_ax = cbar_ax,
                rasterized = True,
                cbar_kws = {'label' : 'rates'},
                cmap = cmap)


