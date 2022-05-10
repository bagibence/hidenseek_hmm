from hidenseek.util.postproc import get_median_time_points_for_every_session

from hidenseek.util.plotting import (
        change_ms_labels_to_sec,
        plot_states_xr,
        show_time_point_names,
        add_vertical_lines_for_time_points
)

def make_fig_stretched_states_in_one_session(session, ax_seek, ax_hide, K, bin_length, orig=True):
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)
    
    stretched_states_seek, stretched_states_hide = session.get_stretched_states(bin_length, median_time_points_seek_bo, median_time_points_hide, orig=orig)
    
    #median_time_points_seek_bo_ji = session.get_time_points('seek').dropna().median()
    #median_time_points_seek_bo_ji['interaction'] = session.get_time_points('seek').median()['interaction']
    median_time_points_seek_bo_ji = median_time_points_seek_bo
    
    plot_states_xr(ax_seek, stretched_states_seek, K, 200, cbar = False)
    plot_states_xr(ax_hide, stretched_states_hide, K, 100, cbar = False)

    # having bin_length here is a bit hacky
    add_vertical_lines_for_time_points(ax_seek, median_time_points_seek_bo / bin_length, 'black', linewidth = 2)
    #show_time_point_names(ax_seek, median_time_points_seek_bo_ji[:-1] / bin_length, ['start', 'box open', 'jump out', 'interaction', 'transit', 'jump in'])

    add_vertical_lines_for_time_points(ax_hide, median_time_points_hide / bin_length, 'black', linewidth = 2)
    #show_time_point_names(ax_hide, median_time_points_hide[1:-1] / bin_length, ['jump out', 'interaction', 'transit', 'jump in'])
    
    change_ms_labels_to_sec(ax_seek, bin_length)
    change_ms_labels_to_sec(ax_hide, bin_length)
    ax_seek.set_xlabel('time (s)')
    ax_hide.set_xlabel('time (s)')

    ax_seek.set_ylabel('trial | seek')
    ax_hide.set_ylabel('trial | hide')
    ax_seek.set_yticks([])
    ax_hide.set_yticks([])
