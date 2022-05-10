import numpy as np
import xarray as xr

import matplotlib.pyplot as plt 
import seaborn as sns

from .colors import get_tab20_and_norm
from .misc import *
from .time_points import *

from hidenseek.util.postproc import get_stretched_state_probs_between_time_points_in_session, get_state_probs_between_time_points_in_session

import warnings


def plot_trial_states(ax, states, K, cbar_ax=None, heatmap_kwargs=None):
    """
    Plot the states of a single trial across time as a heatmap colored by state identity

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    states : 1D array-like
        array contatining the active state across time
    K : int
        number of states
    cbar_ax : plt.Axes (default None)
        Axes to draw the colorbar on
        if None, don't draw a colorbar
    heatmap_kwargs : dict, optional
        keyword arguments to pass to sns.heatmap
    """
    tab20, norm = get_tab20_and_norm(K)

    if isinstance(states, xr.DataArray):
        states_plot_vals = states.to_dataframe('name').T
    else:
        states_plot_vals = np.array(states)[np.newaxis]
        
    if cbar_ax is not None:
        cbar = True
    else:
        cbar = False

    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    
    sns.heatmap(states_plot_vals,
                ax = ax, cmap = tab20, norm = norm, alpha = 0.8,
                rasterized = True,
                cbar = cbar, cbar_ax = cbar_ax, **heatmap_kwargs)


def plot_trial_states_linear(axi, trial, max_time, K, lw=4, mark_extra_points=True):
    """
    Plot the most likely states in a trial on an axis that has length max_time

    Parameters
    ----------
    axi : plt.Axes
        Axes to plot on
    trial : Trial
        trial for which to plot the states
        has to contain the states in trial.states
    max_time : float
        length of the longest trial in the session
        sets the time axis limit
    K : int
        number of states
    lw : int
        linewidth to use for the verical lines
    mark_extra_points : bool, default True
        add white lines for the time points that are
        not signaling the phases of the game

    Returns
    -------
    im : ??
        image plotted on axi by plt.pcolorfast
    """
    tab20, norm = get_tab20_and_norm(K)

    axi.set_xlim((0, max_time))

    im = axi.pcolorfast((0, trial.time_points.end), axi.get_ylim(),
                        trial.states.values[np.newaxis],
                        cmap = tab20, norm = norm,
                        rasterized = True,
                        alpha = 0.8)

    if mark_extra_points:
        add_vertical_lines_for_time_points(axi, trial.all_time_points, 'white', lw)
    add_vertical_lines_for_time_points(axi, trial.time_points.values, 'black', lw)

    return im


def states_xr_to_df(states):
    """
    Convert states from DataArray to DataFrame for plotting

    Parameters
    ----------
    states : xr.DataArray
        some_dim x time DataArray containing the active state

    Returns
    -------
    pd.DataFrame with the first dimension as index and time as columns
    """
    # make sure there is only one dimension that is not time
    # and figure out the name of it
    non_time_dims = [dim for dim in states.dims if dim != 'time']
    assert len(non_time_dims) == 1
    non_time_dim = non_time_dims[0]

    return (states.to_dataframe('name')
                  .reset_index()
                  .pivot(index = non_time_dim, columns = 'time', values = 'name'))


def plot_states_xr(ax, states, K, xticklabels='auto', cbar=True, cbar_ax=None):
    """
    Plot the active states stored in a 2D DataArray

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    states : xr.DataArray
        some_dim x time DataArray containing the state active in every time point 
    K : int
        number of possible states
        for the coloring
    xticklabels : "auto" (default), bool, list-like, or int
        xticklabels argument of seaborn.heatmap
        If True, plot the column names of the dataframe. If False, don’t plot the column names.
        If list-like, plot these alternate labels as the xticklabels.
        If an integer, use the column names but plot only every n label.
        If “auto”, try to densely plot non-overlapping labels.
    cbar : bool, default True
        whether to draw a colorbar
    cbar_ax : plt.Axes (default None)
        Axes to draw the colorbar on
    """
    tab20, norm = get_tab20_and_norm(K)

    if cbar_ax is not None:
        cbar = True

    non_time_dims = [dim for dim in states.dims if dim != 'time']
    assert len(non_time_dims) == 1 or len(non_time_dims) == 0
    if len(non_time_dims) == 1:
        non_time_dim = non_time_dims[0]
    else:
        non_time_dim = 'state'
        states = states.expand_dims('state')

    sns.heatmap(states_xr_to_df(states),
                cmap = tab20, norm = norm,
                rasterized = True,
                alpha = 0.8,
                xticklabels = xticklabels,
                cbar = cbar,
                cbar_ax = cbar_ax,
                cbar_kws = dict(ticks = range(K)),
                ax = ax)
    for i in range(len(getattr(states, non_time_dim))):
        ax.axhline(y = i, color='white')
    ax.set(ylabel = non_time_dim, xlabel = 'time')


def plot_state_probs(ax, big_ax, state_hist, median_time_points, smoothing_length, bin_length, share_y=True, xtick_step=5000, linewidth=2):
    """
    Plot the probability of a given state appearing at a given time
    as these colored density plots below each other

    Parameters
    ----------
    ax : list of Axes
        Axes to plot on
    big_ax : matplotlib Axes
        Axes that wraps around all the others
        put the ylabel on this axis
    state_hist : xr.DataArray
        state x time DataArray containing the probability of a state occurring at at a given time
    median_time_points : array-like
        median times of the phase transitions of the game
    smoothing_lenth : float
        length of the window (in ms) with which to calculate a rolling average
    bin_length : float
        original spike binning window length in ms
    share_y : bool (default True)
        share y axis for the states
    xtick_step : float
        interval of the xticks in ms
    linewidth : float, default 2
        linewidth for the vertical lines
    """

    K = len(state_hist.state)
    roller = int(smoothing_length / bin_length)
    tab20, norm = get_tab20_and_norm(K)
    for s in range(K):
        plot_data = state_hist.sel(state=s).rolling(time=roller, center = True).mean()
        ax[s].plot(plot_data.time, plot_data.values, color = tab20(norm(s)))
        ax[s].fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = tab20(norm(s)), alpha = 0.5)
        ax[s].set_ylabel(f'{s}')
        add_vertical_lines_for_time_points(ax[s], median_time_points, 'black', linewidth = linewidth)
        
        ax[s].set_xlim((0, len(plot_data.time)))
        ax[s].margins(x = 0)
        sns.despine(ax=ax[s], top=True, right=True)

    if share_y:
        ax[0].get_shared_y_axes().join(*ax)
        for axi in ax:
            axi.autoscale()

    ax[0].get_shared_x_axes().join(*ax)
    for axi in ax:
        axi.autoscale()

    for axi in ax:
        if axi != ax[-1]:
            axi.set_xticks([])
        axi.set_yticks([])
    ax[-1].set_xlabel('time')

    # put fewer xticks and rotate the labels
    ax[-1].set_xticks(np.arange(0, plot_data.time[-1], xtick_step))
    ax[-1].set_xticklabels(np.arange(0, plot_data.time[-1], xtick_step).astype(int), rotation=0)

    # put shared ylabel
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('state #')
