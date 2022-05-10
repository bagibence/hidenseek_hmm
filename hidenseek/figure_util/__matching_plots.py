from hidenseek.util.postproc import get_median_time_points_for_every_session

from hidenseek.util.plotting import (
        change_ms_labels_to_sec,
        plot_states_xr,
        show_time_point_names,
        add_vertical_lines_for_time_points
)

import xarray as xr

from hidenseek.db_interface import *
import hidenseek.globals

def make_fig_example_matching(ax_example_matching, sessions, K, bin_length):
    """
    Make figure showing the results of the state matching procedure

    Parameters
    ----------
    ax_example_matching : array of Axes
        subplots on which to plot
    sessions : list of Session
        sessions whose stretched states (in seek) to plot
    K : int
        number of states
    bin_length : int
        bin length in ms

    Returns
    -------
    None
    """
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)
    try:
        iterator = iter(ax_example_matching)
    except TypeError:
        ax_example_matching = [ax_example_matching]
        
    for session_a, axi in zip(sessions, ax_example_matching):
        stretched_states_seek_a, stretched_states_hide_a = session_a.get_stretched_states(bin_length, median_time_points_seek_bo, median_time_points_hide)

        plot_states_xr(axi, stretched_states_seek_a, K, 200, cbar = False)

        add_vertical_lines_for_time_points(axi, median_time_points_seek_bo / bin_length, 'black', linewidth = 2)

        axi.set_ylabel('trial | seek')
        axi.set_yticks([])
        axi.set_title(f'session #{session_a.paper_id}')
        if axi is ax_example_matching[-1]:
            change_ms_labels_to_sec(axi, bin_length)
            axi.set_xlabel('time (s)')
        else:
            axi.set_xticks([])
            axi.set_xlabel('')
