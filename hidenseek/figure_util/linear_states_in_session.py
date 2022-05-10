import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from hidenseek.util.plotting import (
        get_tab20_and_norm,
        share_axis,
        plot_trial_states_linear,
        change_ms_labels_to_sec
    )

def make_fig_linear_states_in_session(session, ax_seek, ax_hide, ax_cbar, big_ax, K, mark_extra_points=True):
    """
    Make figure showing HMM states on all trials of a session without stretching them in time

    Parameters
    ----------
    session : Session
        session whose trials to plot
    ax_seek : array of Axes
        subplots to plot seek trials on
    ax_hide : array of Axes
        subplots to plot hide trials on
    ax_cbar : Axes
        axis for the colorbar for the HMM states
    big_ax : Axes
        axis for the 'trial' ylabel    
    K : int
        number of HMM states
    mark_extra_points : bool, default True
        True: show every time point
        False: only show the game-phase points

    Returns
    -------
    None
    """
    tab20, norm = get_tab20_and_norm(K)
    max_time_seek = np.max([t.time_points.end for t in session.seek_trials])
    max_time_hide = np.max([t.time_points.end for t in session.hide_trials])

    # plot seek trials on the left
    for i, (trial, axi) in enumerate(zip(session.seek_trials, ax_seek)):
        im = plot_trial_states_linear(axi, trial, max_time_seek, K, lw = 1.5, mark_extra_points=mark_extra_points)
        
        if not trial.successful:
            axi.set_ylabel('F', rotation='horizontal', va='center', labelpad=10)
        else:
            axi.set_ylabel('')

    # plot hide trials on the right
    for i, (trial, axi) in enumerate(zip(session.hide_trials, ax_hide)):
        im = plot_trial_states_linear(axi, trial, max_time_hide, K, lw = 1.5, mark_extra_points=mark_extra_points)
        
        if not trial.successful:
            axi.set_ylabel('F', rotation='horizontal', va='center', labelpad=10)
        else:
            axi.set_ylabel('')

    # write seek and hide
    ax_seek[0].set_title('seek')
    ax_hide[0].set_title('hide')
    
    # share x axes in the columns
    share_axis(ax_seek, 'x', -1)
    share_axis(ax_hide, 'x', -1)
    
    # add time label to the bottom
    change_ms_labels_to_sec(ax_seek[-1], 1)
    change_ms_labels_to_sec(ax_hide[-1], 1)
    ax_seek[-1].set_xlabel('time (s)');
    ax_hide[-1].set_xlabel('time (s)');
    
    # remove spines and ticks
    for axi in ax_seek + ax_hide:
        sns.despine(ax=axi, top=True, bottom=True, left=True, right=True)
        axi.set_yticks([])
    
    # put colorbar on the right
    cb = plt.colorbar(im, ticks=norm.boundaries+0.5, cax=ax_cbar)
    cb.set_label('states')
    ax_cbar.set_yticklabels(np.arange(K))
    
    # put y label on the whole thing
    big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    big_ax.set_ylabel('trial')
