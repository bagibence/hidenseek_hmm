# This notebook creates Figure 4.
#
# The images it loads are not included.

# +
import os

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# +
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns

from tqdm.auto import tqdm
# -

# # Load results 

# +
from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

K = 11
transitions = 'sticky'
n_seeds = 40

_, bin_length = load_results(K, transitions, n_seeds=n_seeds)
add_behavioral_states()

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)
# -

# # Set source 

# +
every_nth_frame = 2

figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
fig_source_dir = os.path.join(figures_root_dir, 'states_in_one_image', f'GPFA_{K}_states_{n_seeds}_seeds')

def _scale(im):
    if np.any(im > 2):
        return im / 255
    else:
        return im
    
def load_im(session_id, state_id, method):
    session_dir = os.path.join(fig_source_dir, f"session_{session_id}")
    source_dir = os.path.join(session_dir, f'by_{method}')
        
    im_path = os.path.join(source_dir, f"state_{state_id}_every_{every_nth_frame}_frame.npz")
       
    return _scale(np.rot90(np.load(im_path)['im'], 3))


# -

# # Find transit states 

# +
import hidenseek.globals
from hidenseek.util.postproc import contingency_xr

transit_state_tuples = []
for sid in [13, 12, 9]:
    session = Session[sid]
    trans_state = (contingency_xr(session.behavioral_states, session.states, True)
                   .sel(behavioral_state = hidenseek.globals.behavioral_state_dict['transit'])
                   .argmax()
                   .item()
                  )
    transit_state_tuples.append((sid, trans_state))

# +
transit_method = 'cont'

transit_images = [load_im(session_id, state_id, transit_method) for (session_id, state_id) in transit_state_tuples]

# +
from matplotlib.colors import ListedColormap

transit_state_id = transit_state_tuples[0][1]
transit_cmap = ListedColormap(['white', state_colors[transit_state_id]])
# -

# # Find darting state in the reference session 

# +
darting_sid = 13
session = Session[darting_sid]

darting_k = (contingency_xr(session.behavioral_states, session.states, True)
             .sel(behavioral_state = hidenseek.globals.behavioral_state_dict['darting'])
             .argmax()
             .item()
            )

darting_im = load_im(darting_sid, darting_k, 'position')
# -

# # Load wall- and peeking out states 

# +
wall_sid, wall_k = 13, 9
peek_sid, peek_k = 13, 6

wall_im = load_im(wall_sid, wall_k, 'position')
peek_im = load_im(peek_sid, peek_k, 'position')
# -

# # Match states 

# +
from hidenseek.util.postproc import get_median_time_points_for_every_session

# calculate median time points aggregating every session
median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)

ultimate_med_tp = pd.DataFrame([trial.time_points for trial in Trial.select()]).drop(columns = 'box_open').median()
# -

# add state histograms to the sessions
for session in tqdm(Session.select()):
    session.state_hist_seek, session.state_hist_hide = session.get_state_hist(bin_length, K, median_time_points_seek_bo, median_time_points_hide)
    session.state_probs_seek, session.state_probs_hide = session.get_state_probs(bin_length, median_time_points_seek_bo, median_time_points_hide)
    session.state_hist_seek_dir = session.state_probs_seek.mean('trial')
    session.state_hist_hide_dir = session.state_probs_hide.mean('trial')
    
    session.state_hist_seek['time'], session.state_hist_hide['time'] = session.state_probs_seek.time, session.state_probs_hide.time

# +
from hidenseek.util.postproc.state_matching import match_states_to_reference

# match the states of the different sessions to the reference session
session_a = Session[13]
smoothing_len = 5
use_most_likely = True
matching_method = 'hungarian'

reference_correlations = match_states_to_reference(session_a, smoothing_len, use_most_likely, matching_method, median_time_points_seek_bo, median_time_points_hide, True)
# -

# # Helper functions

# +
from hidenseek.util.plotting import (
    add_vertical_lines_for_time_points,
    change_ms_labels_to_sec,
    show_time_point_names,
    label_subfigures
)

import hidenseek.globals

# +
import matplotlib.patches

def connect_axes(ax1, ax2):
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    
    patch_lower = matplotlib.patches.ConnectionPatch(
        xyA = (1., 0.),
        xyB = (0., 0.),
        coordsA = 'axes fraction',
        coordsB = 'axes fraction',
        axesA = ax1,
        axesB = ax2,
        #color = 'tab:orange',
        color = 'black',
        clip_on = False)

    patch_upper = matplotlib.patches.ConnectionPatch(
        xyA = (1., 1.),
        xyB = (0., 1.),
        coordsA = 'axes fraction',
        coordsB = 'axes fraction',
        axesA = ax1,
        axesB = ax2,
        #color = 'tab:orange',
        color = 'black',
        clip_on = False)

    ax1.add_patch(patch_lower)
    ax1.add_patch(patch_upper)


# -

def connect_inset(ax1, pos1, ax2, pos2):
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    
    patch_lower = matplotlib.patches.ConnectionPatch(
        xyA = (pos1[0], pos1[1]),
        xyB = (pos2[0], pos2[1]),
        coordsA = 'data',
        coordsB = 'axes fraction',
        axesA = ax1,
        axesB = ax2,
        #color = 'tab:orange',
        color = 'black',
        clip_on = False)

    patch_upper = matplotlib.patches.ConnectionPatch(
        xyA = (pos1[2], pos1[3]),
        xyB = (pos2[2], pos2[3]),
        coordsA = 'data',
        coordsB = 'axes fraction',
        axesA = ax1,
        axesB = ax2,
        #color = 'tab:orange',
        color = 'black',
        clip_on = False)

    ax1.add_patch(patch_lower)
    ax1.add_patch(patch_upper)


from hidenseek.util.plotting import plotting_setup
from hidenseek.util.plotting.video import *

# # Make figure 

# +
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = False, tight_layout = False)

wr = [1, 1.625]

gs_up = gridspec.GridSpec(nrows = 1, ncols = 2, width_ratios = wr)
gs_lower = gridspec.GridSpec(nrows = 2, ncols = 2, width_ratios = [1, 1], hspace = 0., wspace = 0.8)
gs_transit_state_label = gridspec.GridSpec(nrows = 2, ncols = 2, width_ratios = [1, 1], hspace = 0., wspace = 0.8)

gs_transit_left = gs_up[0].subgridspec(nrows = 5, ncols = 1, hspace = 0.5)
transit_state_axes = [fig.add_subplot(gs_transit_left[i+1]) for i in range(3)]

gs_transit_right = gs_up[1].subgridspec(nrows = 3, ncols = 1, hspace = 0.1)
transit_im_axes = [fig.add_subplot(gs_transit_right[i], xticks = [], yticks = [], frame_on = True) for i in range(3)]

for (session, axi) in zip([Session[13], Session[12], Session[9]],
                          transit_state_axes):
    ss_seek, ss_hide = session.get_stretched_states(bin_length, median_time_points_seek_bo, median_time_points_hide)
    (ss_seek == transit_state_id).plot(cmap = transit_cmap, ax = axi, add_colorbar=False,
                                       rasterized = True)
    add_vertical_lines_for_time_points(axi, median_time_points_seek_bo, 'black', linewidth=1)
    axi.set_ylabel('trial | seek')
    axi.set_title(f'session #{session.paper_id}')
    axi.set_yticks([])
    change_ms_labels_to_sec(axi)
    axi.set_xlabel('warped time (s)')
    
for axi in transit_state_axes[:-1]:
    axi.set_xlabel('')
    axi.set_xticks([])
    
for (tim, tax) in zip(transit_images, transit_im_axes):
    plot_im_on_ax(tim, tax, state_colors[transit_state_id])
    
ax_transit_time_points = fig.add_subplot(gs_transit_left[1], yticks = [], xticks = [], frame_on = False)
ax_transit_time_points.set_xlim(transit_state_axes[0].get_xlim())
show_time_point_names_same_axis(ax_transit_time_points, median_time_points_seek_bo[:-1], hidenseek.globals.time_point_names_bo_plot[:-1])
ax_transit_time_points.xaxis.set_tick_params(pad = 2*plotting_setup.font_size)

for (ax_state, ax_im) in zip(transit_state_axes, transit_im_axes):
    connect_axes(ax_state, ax_im)

## lower part of the figure
ax_darting_label = fig.add_subplot(gs_lower[0, 0], xticks = [], yticks = [], frame_on = False, label = 'darting_label')
ax_peek_label = fig.add_subplot(gs_lower[0, 1], xticks = [], yticks = [], frame_on = False, label = 'peek_label')
ax_wall_label = fig.add_subplot(gs_lower[1, 0], xticks = [], yticks = [], frame_on = False, label = 'wall_label')

gs_darting = gs_lower[0, 0].subgridspec(nrows = 2, ncols = 1, height_ratios = [1, 7], hspace = 0.05)
gs_wall = gs_lower[1, 0].subgridspec(nrows = 2, ncols = 1, height_ratios = [1, 7], hspace = 0.05)
gs_peek = gs_lower[0, 1].subgridspec(nrows = 2, ncols = 1, height_ratios = [1, 7], hspace = 0.05)

roller = int(1000 / bin_length)

# darting
ss_seek, ss_hide = Session[darting_sid].get_stretched_states(bin_length, ultimate_med_tp, ultimate_med_tp, orig=True)

ax_darting = fig.add_subplot(gs_darting[1], xticks = [], yticks = [], frame_on = True)
ax_darting_probs = fig.add_subplot(gs_darting[0], xticks = [], yticks = [], frame_on = False)
plot_im_on_ax(darting_im, ax_darting, state_colors[darting_k])

plot_data = (xr.concat([ss_seek, ss_hide], dim = 'trial') == darting_k).mean('trial').rolling(time=roller, center=True).mean()
ax_darting_probs.plot(plot_data.time, plot_data.values, color = state_colors[darting_k])
ax_darting_probs.fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = state_colors[darting_k], alpha = 0.5)
add_vertical_lines_for_time_points(ax_darting_probs, ultimate_med_tp, 'black', linewidth = 1)
ax_darting_probs.set_ylabel('P(s)', rotation=0, y = 0.2)


# wall
ss_seek, ss_hide = Session[wall_sid].get_stretched_states(bin_length, ultimate_med_tp, ultimate_med_tp, orig=True)
ax_wall = fig.add_subplot(gs_wall[1], xticks = [], yticks = [], frame_on = True)
ax_wall_probs = fig.add_subplot(gs_wall[0], xticks = [], yticks = [], frame_on = False)
plot_im_on_ax(wall_im, ax_wall, state_colors[wall_k])

plot_data = (xr.concat([ss_seek, ss_hide], dim = 'trial') == wall_k).mean('trial').rolling(time=roller, center=True).mean()
ax_wall_probs.plot(plot_data.time, plot_data.values, color = state_colors[wall_k])
ax_wall_probs.fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = state_colors[wall_k], alpha = 0.5)
add_vertical_lines_for_time_points(ax_wall_probs, ultimate_med_tp, 'black', linewidth = 1)
ax_wall_probs.set_ylabel('P(s)', rotation=0, y = 0.2)

# peek
ss_seek, ss_hide = Session[peek_sid].get_stretched_states(bin_length, ultimate_med_tp, ultimate_med_tp, orig=False)

ax_peek = fig.add_subplot(gs_peek[1], xticks = [], yticks = [], frame_on = True)
ax_peek_probs = fig.add_subplot(gs_peek[0], xticks = [], yticks = [], frame_on = False)
plot_im_on_ax(peek_im, ax_peek, state_colors[peek_k])

plot_data = (xr.concat([ss_seek, ss_hide], dim = 'trial') == peek_k).mean('trial').rolling(time=roller, center=True).mean()
ax_peek_probs.plot(plot_data.time, plot_data.values, color = state_colors[peek_k])
ax_peek_probs.fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = state_colors[peek_k], alpha = 0.5)
#add_vertical_lines_for_time_points(ax_peek_probs, median_time_points_hide, 'black', linewidth = 1)
add_vertical_lines_for_time_points(ax_peek_probs, ultimate_med_tp, 'black', linewidth = 1)
ax_peek_probs.set_ylabel('P(s)', rotation=0, y = 0.2)

ax_darting.set_ylabel(f'running')
ax_peek.set_ylabel(f'peeking')
ax_wall.set_ylabel(f'wall')

ax_transit_state_label = fig.add_subplot(gs_transit_state_label[0, 0], xticks = [], yticks = [], frame_on = False) 
#ax_transit_state_label = fig.add_subplot(gs_up[:], xticks = [], yticks = [], frame_on = False) 

height_ratio = 0.52
#gs_up.tight_layout(fig, rect = [0.125, 0.5, 1., 1.])
#gs_lower.tight_layout(fig, rect = [0.025, 0., 1., 0.5])
gs_up.tight_layout(fig, rect = [0.125, height_ratio, 1., 1.])
gs_lower.tight_layout(fig, rect = [0.025, 0., 1., height_ratio])
gs_transit_state_label.tight_layout(fig, rect = [0.025, 0.5, 1., 0.975])

# show inset
gs_inset = gs_lower[1, 1].subgridspec(nrows = 6, ncols = 3)
ax_inset1 = fig.add_subplot(gs_inset[1:5, 0], xticks = [], yticks = [], frame_on = False)
ax_inset2 = fig.add_subplot(gs_inset[1:5, 1], xticks = [], yticks = [], frame_on = False)
ax_inset3 = fig.add_subplot(gs_inset[1:5, 2], xticks = [], yticks = [], frame_on = False)

def add_peek_inset(top, bottom, left, right, target_ax):
    height = bottom - top
    width = right - left
    target_ax.imshow(peek_im[top:bottom, left:right])

    rect = matplotlib.patches.Rectangle((left, top), width, height, transform = ax_peek.transData, fill=False)
    ax_peek.add_patch(rect)
    connect_inset(ax_peek, (left, bottom, right, bottom), target_ax, (0, 1, 1, 1))
    
add_peek_inset(1050, 1250,
               550, 750,
               ax_inset1)
add_peek_inset(800, 1000,
               750, 950,
               ax_inset2)
add_peek_inset(450, 650,
               800, 1000,
               ax_inset3)

# labels
annotations = label_subfigures([ax_transit_state_label, ax_darting_label, ax_wall_label, ax_peek_label], [0.05, -0.05, -0.05, -0.05])
fig.align_ylabels([ax_transit_state_label, ax_darting_label, ax_wall_label])

# show time points on darting
show_time_point_names_same_axis(ax_darting_probs, ultimate_med_tp, hidenseek.globals.time_point_names_plot)

fig.subplots_adjust(left = 0.0, right = 1., bottom = 0.0, top = 1.)
# -

fig.savefig(os.path.join(figures_root_dir, f'video_fig_with_peek_inset_{K}.png'), dpi = 400)
fig.savefig(os.path.join(figures_root_dir, f'video_fig_with_peek_inset_{K}.pdf'), dpi = 400)


