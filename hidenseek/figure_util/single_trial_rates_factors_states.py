import numpy as np

from hidenseek.util.plotting import (
        get_tab20_and_norm,
        plot_rasterlike_rates,
        plot_trial_states,
        add_vertical_lines_for_time_points,
        change_ms_labels_to_sec,
        show_time_point_names,
        )

import hidenseek.preprocessing as preproc


def make_fig_trial_rates_factors_states(trial, ax_trial_rates, ax_trial_factors, ax_trial_states, ax_trial_cbar_rates, ax_trial_cbar_factors, K, bin_length, rates_cmap=None):
    """
    Make the subplot demonstrating the analysis approach on a single trial

    Parameters
    ----------
    trial : Trial
        trial to plot
    ax_trial_rates : Axes
        axis to plot the firing rates on
    ax_trial_factors : Axes
        axis to plot the factors on
    ax_trial_states : Axes
        axis to plot the HMM states on
    ax_trial_cbar_rates : Axes
        axis for the colorbar of the rates
    ax_trial_cbar_factors : Axes
        axis for the colorbar of the factors
    K : int
        number of HMM states used
    bin_length : int
        bin length in ms
    rates_cmap : colormap or str
        colormap to use for the firing rates

    Returns
    -------
    ticklabels showing the time point names on top of the plot
    """
    tab20, norm = get_tab20_and_norm(K)
    T = len(trial.states.time)
    sample_length = 5
    orig_trial_rates = trial.get_smooth_rates(sample_length, preproc.norm_gauss_window(sample_length, preproc.hw_to_std(250)))
    
    plot_rasterlike_rates(ax_trial_rates, orig_trial_rates, cbar_ax = ax_trial_cbar_rates, cmap = rates_cmap)
    plot_trial_states(ax_trial_states, trial.states, K, cbar_ax = None, heatmap_kwargs = dict(xticklabels = 200))

    trial.factors.plot(ax = ax_trial_factors, cbar_ax=ax_trial_cbar_factors,
                       rasterized = True)
    
    add_vertical_lines_for_time_points(ax_trial_rates, trial.all_time_points / sample_length, 'white', 2)
    add_vertical_lines_for_time_points(ax_trial_rates, trial.time_points / sample_length, 'black', 2)
    add_vertical_lines_for_time_points(ax_trial_states, trial.all_time_points / bin_length, 'white', 2)
    add_vertical_lines_for_time_points(ax_trial_states, trial.time_points / bin_length, 'black', 2)
    add_vertical_lines_for_time_points(ax_trial_factors, trial.all_time_points, 'white', 2)
    add_vertical_lines_for_time_points(ax_trial_factors, trial.time_points, 'black', 2)

    n = orig_trial_rates.neuron.size
    ax_trial_rates.set_ylabel('neuron')
    ax_trial_rates.set_yticks([0, n-1])
    ax_trial_rates.set_yticklabels(np.array([0, n-1]), rotation=0)
    ax_trial_rates.xaxis.set_visible(False)

    d = trial.factors.factor.size
    ax_trial_factors.set_ylabel('factor')
    ax_trial_factors.set_yticks([0, d-1])
    ax_trial_factors.set_yticklabels(np.array([0, d-1]), rotation=0)
    ax_trial_factors.xaxis.set_visible(False)

    ax_trial_states.set_ylabel('state')
    ax_trial_states.set_yticks([])
    #ax_trial_states.set_yticklabels([' '])
    ax_trial_states.set_xlabel('time (s)')

    ax_trial_rates.xaxis.set_visible(False)
    change_ms_labels_to_sec(ax_trial_states, bin_length)

    ax2 = show_time_point_names(ax_trial_rates, trial.time_points.values[:-1] / sample_length, ['start', 'box open', 'jump out', 'interaction', 'transit', 'jump in'], labelrotation=60)
    
    return ax2.get_xticklabels()


