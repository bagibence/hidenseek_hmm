# # Load and set things

# +
import pandas as pd
import os

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# +
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

import matplotlib.transforms as mtransforms
# -

figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
dest_fig_dir = figures_root_dir

# # Load factors and states

# +
from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

K = 11
transitions = 'sticky'

_, bin_length = load_results(K, transitions)

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)
# -

add_behavioral_states()

# # Contingency matrix 

# +
ref_session_num = 13

session = Session[ref_session_num]

# +
from hidenseek.figure_util.fake_state_functions import *
from hidenseek.figure_util.contingency import *

N = 10000
p_value = 5 / 100

correction = 'holm'

# +
# same seed as in generate_MI_scores
seed = 123

generate_fake_states(generate_fake_states_by_rearranging_states, N, seed)
# -

for trial in Trial.select():
    trial.playing_states = trial.behavioral_states

cont_df, cont_df_hmm, cont_df_beh, signif_cont_df, signif_cont_df_hmm, signif_cont_df_beh = make_cont_dfs(session,
                                                                                                          'playing',
                                                                                                          'playing',
                                                                                                          p_value,
                                                                                                          rename = True,
                                                                                                          correction = correction)

# # Darting

behavior = 'darting'
before_pad = 1000
after_pad = 1000

# # Stretched states 

from hidenseek.util.postproc import get_median_time_points_for_every_session

# +
# calculate median time points aggregating every session
median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)

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

# # Make the figure 

# +
from hidenseek.util.plotting import plot_state_probs
from hidenseek.util.postproc import get_stretched_state_probs_between_time_points_in_session
from hidenseek.util.plotting import label_subfigures

from hidenseek.figure_util.contingency_plotting import *
from hidenseek.figure_util.__plot_stretched_states import make_fig_stretched_states_in_one_session
from hidenseek.figure_util.__matching_plots import *

from hidenseek.globals import time_point_names_bo_plot, time_point_names_plot

# +
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = False, tight_layout = False)

gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [1, 5])

gs_ref = gs[0].subgridspec(ncols = 2, nrows = 1, wspace = 0.3)
gs_rest = gs[1].subgridspec(ncols = 2, nrows = 1, wspace = 0.3)

gs_left = gs_rest[0].subgridspec(ncols = 1, nrows = 2)
gs_matching = gs_left[0].subgridspec(ncols = 1, nrows = 2)
gs_probs = gs_left[1].subgridspec(ncols = 1, nrows = K)

gs_right = gs_rest[1].subgridspec(ncols = 1, nrows = 2, height_ratios = [4, 1], hspace = 0.3)

gs_cond_big = gs_right[0].subgridspec(ncols = 2, nrows = 1, width_ratios = [20, 1])
gs_cond = gs_cond_big[0].subgridspec(ncols = 1, nrows = 2)

gs_darting_probs = gs_right[1].subgridspec(ncols = 2, nrows = 1, width_ratios = [20, 1])

ax_ref_seek = fig.add_subplot(gs_ref[0])
ax_ref_hide = fig.add_subplot(gs_ref[1])
ax_ref_label = fig.add_subplot(gs_ref[:], frame_on = False, xticks = [], yticks = [])

ax_matching = [fig.add_subplot(gsi) for gsi in gs_matching]
ax_matching_label = fig.add_subplot(gs_matching[:], frame_on = False, xticks = [], yticks = [])

ax_probs = [fig.add_subplot(gsi) for gsi in gs_probs]
ax_probs_label = fig.add_subplot(gs_probs[:], frame_on = False, xticks = [], yticks = [], label='C')
big_ax_probs = fig.add_subplot(gs_probs[:], frame_on = False, xticks = [], yticks = [], label='big_ax')

ax_cond_cbar = fig.add_subplot(gs_cond_big[1])
ax_cond_hmm = fig.add_subplot(gs_cond[0])
ax_cond_beh = fig.add_subplot(gs_cond[1])
ax_cond_label = fig.add_subplot(gs_cond_big[:], frame_on = False, xticks = [], yticks = [])

ax_darting_probs = fig.add_subplot(gs_darting_probs[0])
ax_darting_probs_cbar = fig.add_subplot(gs_darting_probs[1])
ax_darting_probs_label = fig.add_subplot(gs_right[1], frame_on = False, xticks = [], yticks = [])

make_fig_stretched_states_in_one_session(Session[ref_session_num], ax_ref_seek, ax_ref_hide, K, bin_length)
ax_ref_seek.set_title(f"session #{Session[ref_session_num].paper_id}")
ax_ref_hide.set_title(f"session #{Session[ref_session_num].paper_id}")
make_fig_example_matching(ax_matching, [Session[9], Session[12]], K, bin_length)

ax_ref_seek.set_title('')
ax_ref_seek_title = ax_ref_seek.twinx()
ax_ref_seek_title.set_yticks([])
ax_ref_seek_title.set_ylabel(f'session #{Session[ref_session_num].paper_id}')
ax_ref_hide.set_title('')
ax_ref_hide_title = ax_ref_hide.twinx()
ax_ref_hide_title.set_yticks([])
ax_ref_hide_title.set_ylabel(f'session #{Session[ref_session_num].paper_id}')

for (axi, s) in zip(ax_matching, [Session[9], Session[12]]):
    axi.set_title('')
    axi = axi.twinx()
    axi.set_yticks([])
    axi.set_ylabel(f'session #{s.paper_id}')


ax2 = show_time_point_names(ax_ref_seek, median_time_points_seek_bo[:-1] / bin_length, time_point_names_bo_plot[:-1])
show_time_point_names(ax_ref_hide, median_time_points_hide[:-1] / bin_length, time_point_names_plot[:-1])

matched_state_histograms_seek_xr = xr.concat([s.state_hist_seek_dir for s in Session.select()], dim = 'session')
plot_state_probs(ax_probs, big_ax_probs, matched_state_histograms_seek_xr.mean('session'), median_time_points_seek_bo, 500, bin_length, True, xtick_step = 20000, linewidth = 1)
change_ms_labels_to_sec(ax_probs[-1], 1)
for axi in ax_probs:
    axi.set_ylabel('')
big_ax_probs.set_ylabel('HMM state')

for axi in [ax_ref_seek, ax_ref_hide, ax_matching[-1], ax_probs[-1]]:
    axi.set_xlabel("warped time (s)")

make_fig_hmm_behavior_conditional_probs(cont_df_hmm.T, cont_df_beh.T,
                                        ax_cond_hmm, ax_cond_beh, ax_cond_cbar,
                                        state_colors,
                                        signif_cont_df_hmm=signif_cont_df_hmm.T,
                                        signif_cont_df_beh=signif_cont_df_beh.T)
ax_cond_cbar.set_ylabel('Conditional probability')

darting_probs = get_stretched_state_probs_between_time_points_in_session(Session[13], f'{behavior}_start_times', f'{behavior}_end_times', 'both', before_pad, after_pad, behavior).mean(behavior)
darting_probs = darting_probs.assign_coords({'time' : darting_probs.time/1000,
                                             'state' : range(K)})
darting_probs.plot(ax = ax_darting_probs, cbar_ax = ax_darting_probs_cbar, yincrease=False, yticks=darting_probs.state.values, vmin = 0, vmax = 1, cbar_kwargs = {'label' : "P(state | darting)", 'ticks' : [0, 1]},
                   rasterized = True)
ax_darting_probs.axvline(x = before_pad/1000, color = "white")
ax_darting_probs.axvline(x = darting_probs.time.values[-1]-(after_pad/1000), color = "white")
ax_darting_probs.set_xlabel("time (s)")
ax_darting_probs.set_ylabel("HMM state")
#ax_darting_probs.set_title("P(state | darting)")
ax2 = show_time_point_names(ax_darting_probs, [before_pad/1000, darting_probs.time.values[-1]-(after_pad/1000)], ['start', 'end'], 0)
plt.setp(ax2.xaxis.get_majorticklabels(), ha='center')


for axi in [ax_cond_beh, ax_cond_hmm, ax_darting_probs]:
    for t, c in zip(axi.get_yticklabels(), state_colors):
        t.set_color(c)
        t.set_weight('bold')
        t.set_size(6)
        
for axi in [ax_cond_beh, ax_cond_hmm]:
    for text in axi.texts:
        text.set_text(r'$\ast$')

#dh = 0.075
dh = 0.075 * 0.75

pos1 = ax_cond_hmm.get_position()
w1 = pos1.width
h1 = pos1.height
pos2 = ax_cond_beh.get_position()
pos3 = ax_cond_cbar.get_position()
wc = pos3.width
hc = pos3.height

ax_cond_hmm.set_position([pos1.x0, pos1.y0 + dh, w1, h1 - dh])
ax_cond_beh.set_position([pos2.x0, pos2.y0 + 2*dh, w1, h1 - dh])
ax_cond_cbar.set_position([pos3.x0, pos3.y0 + 2*dh, wc, hc - 2*dh])

annotations = label_subfigures([ax_ref_label, ax_matching_label, ax_probs_label, ax_cond_label, ax_darting_probs_label], 'auto', -0.05)

for axi in [ax_cond_hmm, ax_cond_beh]:
    plt.setp(axi.get_yticklabels(), rotation = 0)
# -
fig.savefig(os.path.join(dest_fig_dir, f'Fig3_p_{p_value:.4f}_K_{K}_N_{N}_{correction}_correction_session_side.pdf'), dpi = 400)
fig.savefig(os.path.join(dest_fig_dir, f'Fig3_p_{p_value:.4f}_K_{K}_N_{N}_{correction}_correction_session_side.png'), dpi = 400)


