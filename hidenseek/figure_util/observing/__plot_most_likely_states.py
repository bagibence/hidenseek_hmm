import hidenseek.util.plotting as plotting
import hidenseek.globals
from hidenseek.util.postproc import get_median_time_points_for_every_session

def make_fig_most_likely_states(session, ax_seek_obs, ax_seek_play, ax_hide_obs, ax_hide_play, big_ax_seek, big_ax_hide, K, bin_length):
    """
    Make the figure with the most likely stretched states in observing and playing in an example session

    Parameters
    ----------
    session : Session
        session whose stretched states to plot
    ax_seek_obs : Axes
        subplot for seek trials when observing
    ax_seek_play : Axes
        subplot for seek trials when playing
    ax_hide_obs : Axes
        subplot for hide trials when observing
    ax_hide_play : Axes
        subplot for hide trials when playing
    big_ax_seek : Axes
    big_ax_hide : Axes
        looks like these are not used in the function anymore...
    K : int
        number of states
    bin_length : int
        bin length in ms
        
    Returns
    -------
    array of the time point ticklabels
    """
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)
    
    stretched_states_seek, stretched_states_hide = session.get_stretched_states(bin_length, median_time_points_seek_bo, median_time_points_hide)
    stretched_states_seek['trial'] = [t.id for t in session.seek_trials]
    stretched_states_hide['trial'] = [t.id for t in session.hide_trials]

    seek_obs_trials = [t.id for t in session.seek_trials if t.observing]
    hide_obs_trials = [t.id for t in session.hide_trials if t.observing]
    seek_play_trials = [t.id for t in session.seek_trials if not t.observing]
    hide_play_trials = [t.id for t in session.hide_trials if not t.observing]

    # seek trials
    plotting.plot_states_xr(ax_seek_obs, stretched_states_seek.sel(trial = seek_obs_trials), K, 200, cbar = False)
    plotting.add_vertical_lines_for_time_points(ax_seek_obs, median_time_points_seek_bo / bin_length, 'black', linewidth = 2)

    plotting.plot_states_xr(ax_seek_play, stretched_states_seek.sel(trial = seek_play_trials), K, 200, cbar = False)
    plotting.add_vertical_lines_for_time_points(ax_seek_play, median_time_points_seek_bo / bin_length, 'black', linewidth = 2)

    ax_seek_play.set(xticks = [], xlabel = '')
    plotting.show_time_point_names(ax_seek_play, median_time_points_seek_bo[:-1] / bin_length, hidenseek.globals.time_point_names_bo_plot[:-1])
    plotting.change_ms_labels_to_sec(ax_seek_obs, bin_length)
    ax_seek_obs.set(xlabel = 'time (s)')

    ax_seek_obs.set_ylabel('obs. | seek')
    ax_seek_play.set_ylabel('play | seek')

    # hide trials
    plotting.plot_states_xr(ax_hide_obs, stretched_states_hide.sel(trial = hide_obs_trials), K, 200, cbar = False)
    plotting.add_vertical_lines_for_time_points(ax_hide_obs, median_time_points_hide / bin_length, 'black', linewidth = 2)

    plotting.plot_states_xr(ax_hide_play, stretched_states_hide.sel(trial = hide_play_trials), K, 200, cbar = False)
    plotting.add_vertical_lines_for_time_points(ax_hide_play, median_time_points_hide / bin_length, 'black', linewidth = 2)

    ax_hide_play.set(xticks = [], xlabel = '')
    ax2 = plotting.show_time_point_names(ax_hide_play, median_time_points_hide[:-1] / bin_length, hidenseek.globals.time_point_names_plot[:-1])
    plotting.change_ms_labels_to_sec(ax_hide_obs, bin_length)
    ax_hide_obs.set(xlabel = 'time (s)')

    ax_hide_obs.set_ylabel('obs. | hide')
    ax_hide_play.set_ylabel('play | hide')
    
    for axi in [ax_seek_obs, ax_seek_play, ax_hide_obs, ax_hide_play]:
        axi.set_yticks([])
        
    return ax2.get_xticklabels()
