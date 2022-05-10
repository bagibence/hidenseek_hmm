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


# +
from hidenseek.util.postproc import get_median_time_points_for_every_session

_, med_tp_seek, med_tp_hide = get_median_time_points_for_every_session(True)

ultimate_med_tp = pd.DataFrame([trial.time_points for trial in Trial.select()]).drop(columns = 'box_open').median()

# +
from hidenseek.util.plotting import (
    add_vertical_lines_for_time_points,
    change_ms_labels_to_sec,
    show_time_point_names,
    label_subfigures,
    plot_im_on_ax,
    show_time_point_names_same_axis
)

import hidenseek.globals

# +
interaction_sid, interaction_k, interaction_method = 13, 8, 'cont'

interaction_im = load_im(interaction_sid, interaction_k, interaction_method)
interaction_color = state_colors[interaction_k]

# +
fig = plt.figure(figsize = (cb_width, cb_height / 2), constrained_layout = True)

gs_hand = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [1, 7], hspace = 0.05)

roller = int(1000 / bin_length)

# hand
ss_seek, ss_hide = Session[interaction_sid].get_stretched_states(bin_length, ultimate_med_tp, ultimate_med_tp, orig=False)

ax_hand = fig.add_subplot(gs_hand[1], xticks = [], yticks = [], frame_on = True)
ax_interaction_probs = fig.add_subplot(gs_hand[0], xticks = [], yticks = [], frame_on = False)
plot_im_on_ax(interaction_im, ax_hand, interaction_color)

plot_data = (xr.concat([ss_seek, ss_hide], dim = 'trial') == interaction_k).mean('trial').rolling(time=roller, center=True).mean()
ax_interaction_probs.plot(plot_data.time, plot_data.values, color = interaction_color)
ax_interaction_probs.fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = interaction_color, alpha = 0.5)
#add_vertical_lines_for_time_points(ax_interaction_probs, med_tp_hide, 'black', linewidth = 1)
add_vertical_lines_for_time_points(ax_interaction_probs, ultimate_med_tp, 'black', linewidth = 1)
#ax_interaction_probs.set_ylabel('P(s)')

show_time_point_names_same_axis(ax_interaction_probs, ultimate_med_tp, hidenseek.globals.time_point_names_plot)
# -

# # Success vs fail in seek trials 

session = Session[13]

# +
time_tuples = []
for trial in session.trials:
    result = 'successful' if trial.successful else 'failed'
    trial_length = trial.states.size
    role = trial.role
    for k in range(K):
        time_spent = (trial.states == k).sum().item()
        time_fraction = time_spent / trial_length
        time_tuples.append((k, result, time_spent * bin_length / 1000, time_fraction, role)) 
        
tdf = pd.DataFrame(time_tuples, columns = ('state', 'result', 'time', 'time_fraction', 'role'))
# -

from hidenseek.util.plotting import unique_fig_legend, unique_legend

# +
fig, ax = plt.subplots(figsize = (A4_width / 3 * 2, A4_height / 3))
sns.barplot(x = 'state', y = 'time', hue = 'result', data = tdf.query("role == 'seek'"), dodge = True, ci = None, estimator = np.median, alpha = 0.4, ax = ax)
sns.stripplot(x = 'state', y = 'time', hue = 'result', data = tdf.query("role == 'seek'"), dodge = True, ax = ax)

[t.set_color(tab20(norm(i))) for (i, t) in zip(range(K), ax.xaxis.get_ticklabels())];
[t.set_fontweight('bold') for t in ax.xaxis.get_ticklabels()];

#unique_fig_legend(fig, np.array([ax]), loc = (1., 0.5))
ax.legend(loc = (1.05, 0.5))
# -

# # Combine the two plots 

# +
fig = plt.figure(figsize = (narrow_cb_width, 0.48 * cb_height), constrained_layout = False)

gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [2, 1])
gs_interaction = gs[0].subgridspec(nrows = 2, ncols = 1, height_ratios = [1, 8], hspace = 0.05)
#gs_failed = gs[1].subgridspec(nrows = 1, ncols = 2, width_ratios = [6, 1])
gs_failed = gs[1]

roller = int(1000 / bin_length)

# interaction
ss_seek, ss_hide = Session[interaction_sid].get_stretched_states(bin_length, ultimate_med_tp, ultimate_med_tp, orig=False)

ax_interaction = fig.add_subplot(gs_interaction[1], xticks = [], yticks = [], frame_on = True)
ax_interaction_probs = fig.add_subplot(gs_interaction[0], xticks = [], yticks = [], frame_on = False)
plot_im_on_ax(interaction_im, ax_interaction, interaction_color)

plot_data = (xr.concat([ss_seek, ss_hide], dim = 'trial') == interaction_k).mean('trial').rolling(time=roller, center=True).mean()
ax_interaction_probs.plot(plot_data.time, plot_data.values, color = interaction_color)
ax_interaction_probs.fill_between(plot_data.time, np.zeros_like(plot_data.time), plot_data.values, color = interaction_color, alpha = 0.5)
add_vertical_lines_for_time_points(ax_interaction_probs, ultimate_med_tp, 'black', linewidth = 1)
ax_interaction_probs.set_ylabel('P(s)')

show_time_point_names_same_axis(ax_interaction_probs, ultimate_med_tp, hidenseek.globals.time_point_names_plot)
#show_time_point_names(ax_interaction_probs, ultimate_med_tp, hidenseek.globals.time_point_names_plot)


#ax_failed = fig.add_subplot(gs_failed[0])
ax_failed = fig.add_subplot(gs_failed)
sns.barplot(x = 'state', y = 'time', hue = 'result', data = tdf.query("role == 'seek'"), dodge = True, ci = None, estimator = np.median, alpha = 0.4, ax = ax_failed)
sns.stripplot(x = 'state', y = 'time', hue = 'result', data = tdf.query("role == 'seek'"), dodge = True, ax = ax_failed)
[t.set_color(tab20(norm(i))) for (i, t) in zip(range(K), ax_failed.xaxis.get_ticklabels())];
[t.set_fontweight('bold') for t in ax_failed.xaxis.get_ticklabels()];
ax_failed.set_ylabel('time (sec.)')
sns.despine(ax = ax_failed)

unique_legend(ax_failed, fancybox = False, frameon=False)

ax_interaction_label = fig.add_subplot(gs[0], xticks = [], yticks = [], frame_on = False)
ax_failed_label = fig.add_subplot(gs[1], xticks = [], yticks = [], frame_on = False)

label_subfigures([ax_interaction_label, ax_failed_label], 'auto')

#fig.align_ylabels([ax_interaction_label, ax_failed_label])
#fig.align_ylabels([ax_interaction_probs, ax_failed])
# -

fig.savefig(os.path.join(figures_root_dir, f'supp_fig_interaction_video_and_failed_trials_K_{K}.png'), dpi = 400)
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_interaction_video_and_failed_trials_K_{K}.pdf'), dpi = 400)


