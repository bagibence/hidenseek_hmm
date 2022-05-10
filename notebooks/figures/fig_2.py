# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python (hidenseek)
#     language: python
#     name: hidenseek
# ---

# %% [markdown]
# # Load and set things

# %%
import pandas as pd
import os

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting.colors import parula_map
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

from hidenseek.figure_util.MI_distribution import count_low_chance_scores
from hidenseek.figure_util.linear_states_in_session import *
from hidenseek.figure_util.single_trial_rates_factors_states import *
from hidenseek.figure_util.umap_embedding import embed_with_umap

import matplotlib.transforms as mtransforms

# %%
from hidenseek.figure_util.load_results import load_results

K = 11
transitions = 'sticky'

_, bin_length = load_results(K, transitions)

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
source_fig_dir = os.path.join(figures_root_dir, 'source_images')
dest_fig_dir = figures_root_dir

hmm_path = os.path.join(source_fig_dir, "HMM_diagram.png")
hmm_im = plt.imread(hmm_path)


# %% [markdown]
# # Get ready to plot 

# %% [markdown]
# ## define some functions 

# %%
def make_fig_umap_3d_hide_and_seek(session, ax_seek, ax_hide):
    for trial in session.seek_trials:
        ax_seek.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = [state_colors[k] for k in trial.states.values], s = 0.5)
    for trial in session.hide_trials:
        ax_hide.scatter(trial.embedding[:, 0], trial.embedding[:, 1], trial.embedding[:, 2], color = [state_colors[k] for k in trial.states.values], s = 0.5)

    labelpad = -15
    for ax in [ax_seek, ax_hide]:
        ax.set_xlabel('$x_1$', labelpad = labelpad)
        ax.set_ylabel('$x_2$', labelpad = labelpad)
        ax.set_zlabel('$x_3$', labelpad = labelpad)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ax_seek.set_title('seek')
    ax_hide.set_title('hide')


# %%
def move_ax(axi, up, left):
    ax_box = axi.get_position()
    width = ax_box.x1 - ax_box.x0
    height = ax_box.y1 - ax_box.y0
    axi.set_position([ax_box.x0 - left, ax_box.y0 - up, width, height])


# %%
def resize_ax(axi, plus_height, plus_right):
    ax_box = axi.get_position()
    width = ax_box.x1 - ax_box.x0
    height = ax_box.y1 - ax_box.y0
    axi.set_position([ax_box.x0, ax_box.y0, width+plus_right, height+plus_height])


# %%
from hidenseek.figure_util.signif_code import signif_code
from hidenseek.figure_util.get_p import get_p
from hidenseek.figure_util.observing.__plot_mi import plot_mi

# %% [markdown]
# ## Prepare the data 

# %%
# trial to plot in B
trial_num = 229

session = Session[13]

embed_with_umap(session)

mi_df = pd.read_csv(f'real_and_fake_mi_scores_K_{K}.csv')

# %%
mi_df['paper_id'] = mi_df.apply(lambda row: Session[row.session_id].paper_id, axis = 1)

# %% [markdown]
# # Make the figure 

# %%
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = False, tight_layout = False)

gs = fig.add_gridspec(nrows = 2, ncols = 1, hspace = 0.1, height_ratios = [10, 1])
gs_upper = gs[0].subgridspec(nrows = 3, ncols = 2, height_ratios = [2, 2, 2], width_ratios = [60, 1], hspace = 0.4, wspace = 0.2)

# A and B) HMM diagram and single trial plot with rates and states
trial = Trial[trial_num]

gs_trial              = gs_upper[0, 0].subgridspec(nrows = 3, ncols = 4, height_ratios = [20, 16, 3], width_ratios = [20, 32, 1, 3], wspace = 0.05)
ax_trial_rates        = fig.add_subplot(gs_trial[0, 1])
ax_trial_factors      = fig.add_subplot(gs_trial[1, 1])
ax_trial_states       = fig.add_subplot(gs_trial[2, 1])
ax_trial_cbar_rates   = fig.add_subplot(gs_trial[0, 2])
ax_trial_cbar_factors = fig.add_subplot(gs_trial[1, 2])
ax_trial_label        = fig.add_subplot(gs_trial[:, 1:], frame_on = False, xticks = [], yticks = [])

ax_hmm = fig.add_subplot(gs_trial[:, 0], frame_on=False, xticks = [], yticks = [])
ax_hmm_label = fig.add_subplot(gs_upper[0, :], frame_on = False, xticks = [], yticks = [])

ax_hmm.imshow(hmm_im)
move_ax(ax_hmm, 0.01, 0.08)
resize_ax(ax_hmm, -0.02, 0)

ticklabels = make_fig_trial_rates_factors_states(trial, ax_trial_rates, ax_trial_factors, ax_trial_states, ax_trial_cbar_rates, ax_trial_cbar_factors, K, bin_length,
                                                 rates_cmap = parula_map)

ax_trial_cbar_rates.set_ylabel('rate\n(a.u.)')
ax_trial_cbar_factors.set_ylabel('magnitude\n(a.u.)')

fig.align_ylabels([ax_trial_rates, ax_trial_factors, ax_trial_states])


# C) states in every trial of a session
num_trials_max = max(len(session.seek_trials), len(session.hide_trials))
#gs_states = gs[1].subgridspec(nrows = 1, ncols = 3, width_ratios = [14.5, 14.5, 1])
gs_states = gs_upper[1, 0].subgridspec(nrows = 1, ncols = 2)
gs_seek = gs_states[0].subgridspec(nrows = num_trials_max, ncols = 1)
gs_hide = gs_states[1].subgridspec(nrows = num_trials_max, ncols = 1)
ax_seek = [fig.add_subplot(gs_seek[i]) for i in range(num_trials_max)]
ax_hide = [fig.add_subplot(gs_hide[i]) for i in range(num_trials_max)]
ax_cbar = fig.add_subplot(gs_upper[:, 1])
ax_states_whole = fig.add_subplot(gs_states[:], frame_on = False, xticks = [], label = 'trial')
ax_states_label = fig.add_subplot(gs_upper[1, :], frame_on = False, xticks = [], yticks = [], label = 'C')

make_fig_linear_states_in_session(session, ax_seek, ax_hide, ax_cbar, ax_states_whole, K, mark_extra_points=False)

# D) UMAP in seek and hide separately
gs_umap       = gs_upper[2, 0].subgridspec(nrows = 1, ncols = 2, width_ratios = [1, 1], wspace = 0.2)
ax_seek_umap  = fig.add_subplot(gs_umap[0], projection='3d')
ax_hide_umap  = fig.add_subplot(gs_umap[1], projection='3d')
ax_umap_label = fig.add_subplot(gs_upper[2, :], frame_on = False, xticks = [], yticks = [])

make_fig_umap_3d_hide_and_seek(session, ax_seek_umap, ax_hide_umap)

# E) mutual info plot
gs_lower = gs[1].subgridspec(ncols = 2, nrows = 1, width_ratios = [120, 1])
gs_mutual_info = gs_lower[:].subgridspec(ncols = 2, nrows = 1, wspace = 0.5)
ax_mutual_info = fig.add_subplot(gs_mutual_info[:])
ax_mutual_info_label = fig.add_subplot(gs_mutual_info[:], frame_on = False, xticks = [], yticks = [], label = 'D')

plot_mi(mi_df, ax_mutual_info, scale = 0.5, x_field = "paper_id")
sns.despine(ax = ax_mutual_info)
ax_mutual_info.set_ylabel('MI')


x_offs = -0.1
y_offs = 1.0
annotations = []
for axi, label, x_offset, y_offset in zip([ax_hmm_label, ax_trial_label, ax_states_label, ax_umap_label, ax_mutual_info_label],
                                          ['A', 'B', 'C', 'D', 'E'],
                                          [x_offs, 55/32 * x_offs, x_offs, x_offs, x_offs],
                                          [y_offs, y_offs, y_offs, y_offs*0.8, y_offs]):
    ann = axi.set_ylabel(label, size = subplot_labelsize, weight = 'bold', rotation = 0)
    annotations.append(ann)
    axi.yaxis.set_label_coords(x_offset, y_offset)

fig.align_ylabels([ax_hmm_label, ax_states_label, ax_umap_label, ax_mutual_info_label])

fig.canvas.draw()

## move the plots in A so that the top of the time point names aligns with the top of the A label

# get the height of the ABCD label
ann_box = annotations[0].get_window_extent().transformed(fig.transFigure.inverted())
ann_height = ann_box.y1 - ann_box.y0

# get the height of the time point labels
bboxes = []
for label in ticklabels:
    bbox = label.get_window_extent()
    # the figure transform goes from relative coords->pixels and we want the inverse of that
    bboxi = bbox.transformed(fig.transFigure.inverted())
    bboxes.append(bboxi)

# this is the bbox that bounds all the bboxes, again in relative figure coords
bbox = mtransforms.Bbox.union(bboxes)

# move the state probability axes so that the top of the time point labels
# lines up with the top of the ABCD label
moving_height = bbox.height - ann_height

# actually do the moving
for axi in [ax_hmm, ax_trial_rates, ax_trial_factors, ax_trial_states, ax_trial_cbar_rates, ax_trial_cbar_factors,
            ax_hmm_label, ax_trial_label,
            ax_seek_umap, ax_hide_umap,
            ax_mutual_info,
            ax_mutual_info_label, ax_umap_label,
            *ax_seek, *ax_hide, ax_cbar, ax_states_whole, ax_states_label]:
    ax_box = axi.get_position()
    width = ax_box.x1 - ax_box.x0
    height = ax_box.y1 - ax_box.y0
    axi.set_position([ax_box.x0, ax_box.y0 - moving_height, width, height])
    

fig.align_ylabels([ax_hmm_label, ax_states_label, ax_umap_label, ax_mutual_info_label])

# %%
fig.savefig(os.path.join(dest_fig_dir, f"Fig2_K_{K}.pdf"), dpi = 400)
fig.savefig(os.path.join(dest_fig_dir, f"Fig2_K_{K}.png"), dpi = 400)

# %%
count_low_chance_scores(mi_df)

# %% [markdown]
# # MI as a function of neurons 

# %%
real_df = mi_df.query('kind == "real"').copy()
real_df['n_neurons'] = [Session[sid].recorded_cells.count() for sid in real_df.session_id]

# %%
fig, ax = plt.subplots(figsize = (A4_width / 2, A4_height / 4))
real_df.plot.scatter(x = 'n_neurons', y = 'score', ax = ax)
ax.set_xlabel('Number of neurons')
ax.set_ylabel('MI with behavior')

# %%
