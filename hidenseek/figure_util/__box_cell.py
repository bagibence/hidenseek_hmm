import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from hidenseek.preprocessing import filter_between


def make_fig_box_cell(neur, ax_raster, ax_rate, ax_box_closed, before_pad, after_pad, ymax=1.):
    """
    Make subplot for the cell that responds to the box closed events

    Parameters
    ----------
    neur : Cell
        neuron whose spikes to plot
    ax_raster : Axes
        axis on which to plot the raster per trial
    ax_rate : Axes
        axis on which to plot the trial-averaged rate
    ax_box_closed : Axes
        big axis on which to draw the vertical line at t=0
    before_pad : int
        time padding before events in ms
    after_pad : int
        time padding after events in ms
    ymax : float, default 1.
        ratio of ax_box_closed until the vertical line should go

    Returns
    -------
    None
    """
    neur.bo_spikes = [filter_between(neur.all_spikes, t-before_pad, t+after_pad) - t for t in neur.session.abs_box_closed_times]
    all_bo_spikes = np.concatenate(neur.bo_spikes)
    
    for i, sp in enumerate(neur.bo_spikes):
        ax_raster.scatter(sp / 1000, np.ones_like(sp) * i, color = 'tab:blue', s = 1)

    ax_raster.set_ylabel('event #')
    ax_raster.set_yticks([0, len(neur.bo_spikes)-1])
    ax_raster.set_yticklabels([1, len(neur.bo_spikes)])
    plt.setp(ax_raster.get_xticklabels(), visible = False)
    ax_raster.xaxis.set_tick_params(length = 0)

    sns.histplot(x = all_bo_spikes / 1000, bins = np.arange(-before_pad/1000, (after_pad+1)/1000, 500/1000), weights = np.ones_like(all_bo_spikes)*2/len(neur.bo_spikes), color = 'tab:blue',
                 ax = ax_rate)
    ax_rate.set_xlabel('time (s)')
    ax_rate.set_ylabel('firing rate (Hz)')

    sns.despine(ax = ax_raster, bottom = True)
    sns.despine(ax = ax_rate)
    ax_box_closed.axvline(x = 0, color = 'black', ymax = ymax)

