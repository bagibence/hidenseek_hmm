# +
import os
import sys
from tqdm.auto import tqdm

from hidenseek.db_interface import *
connect_to_db()

from hidenseek.util.plotting.plotting_setup import *

# +
from hidenseek.figure_util.__pooled_rates import *
from hidenseek.figure_util.__box_cell import *

from hidenseek.util.plotting import *
# -

import kneed

figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
source_dir = os.path.join(figures_root_dir, 'source_images')

# # Load things

# +
arena = plt.imread(os.path.join(source_dir, 'arena_with_rat_and_experimenter_circled.png'))
trial_structure = plt.imread(os.path.join(source_dir, 'trial_structure.png'))

arena_horizontal = np.rot90(arena, 3)

# +
before_pad = 5000
after_pad = 5000

neur = Cell[172]

# +
bin_length = 250

pooled_rates_seek, pooled_rates_hide = make_trial_averaged_rates_for_all_sessions(bin_length, 'scale')
# -

idx_seek, idx_hide = hier_cluster_pooled_rates(pooled_rates_seek, pooled_rates_hide)

k = 6
np.random.seed(123)
pooled_labels_seek, pooled_labels_hide = kmeans_pooled_rates(pooled_rates_seek, pooled_rates_hide, k)

# # Make figure 

show_num_neurons = False

# +
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = True)
gs = fig.add_gridspec(nrows = 2, ncols = 1, height_ratios = [1, 1])

gs_upper = gs[0].subgridspec(nrows = 2, ncols = 2, width_ratios = [2, 1], height_ratios = [2, 3])

ax_trial_structure = fig.add_subplot(gs_upper[0, 0], frame_on = False, xticks = [], yticks = [], label = 'trial_structure')
ax_arena = fig.add_subplot(gs_upper[1, 0], frame_on = True, xticks = [], yticks = [], label = 'arena')

gs_box_closed = gs_upper[:, 1].subgridspec(nrows = 2, ncols = 1, height_ratios = [2, 1], hspace = 0.1)
ax_rate = fig.add_subplot(gs_box_closed[1])
ax_raster = fig.add_subplot(gs_box_closed[0], sharex = ax_rate)
ax_box_closed = fig.add_subplot(gs_box_closed[:], frame_on = False, yticks = [], sharex = ax_rate)

gs_pooled = gs[1].subgridspec(nrows = 2, ncols = 3, height_ratios = [10, 1], hspace = 0.1, width_ratios = [20, 20, 1])
ax_pooled = np.array([fig.add_subplot(gsi) for gsi in gs_pooled]).reshape((2, 3))


# axes for the ABC labels
ax_trial_structure_label = fig.add_subplot(gs_upper[0, 0], frame_on = False, xticks = [], yticks = [])
ax_arena_label = fig.add_subplot(gs_upper[1, 0], frame_on = False, xticks = [], yticks = [])
ax_box_closed_label = fig.add_subplot(gs_box_closed[:], frame_on = False, yticks = [], xticks = [])
ax_pooled_label = fig.add_subplot(gs_pooled[:], frame_on = False, yticks = [], xticks = [])


# draw subplots
ax_trial_structure.imshow(trial_structure, extent = [0, trial_structure.shape[1], 0, trial_structure.shape[0]],
                          rasterized = True)

ax_arena.imshow(arena_horizontal, extent = [0, arena_horizontal.shape[1], 0, arena_horizontal.shape[0]],
                rasterized = True)

make_fig_box_cell(neur, ax_raster, ax_rate, ax_box_closed, before_pad, after_pad, ymax=0.97)
ax_box_closed.annotate('box closed', (0.5, 1.0), xycoords = 'axes fraction', horizontalalignment = 'center')

ax2_seek, ax2_hide = make_pooled_neurons_fig(ax_pooled, pooled_rates_seek, pooled_rates_hide, idx_seek, idx_hide, pooled_labels_seek, pooled_labels_hide, bin_length, k, cmap = parula_map, cbar=True)

if show_num_neurons:
    ax_pooled[0][0].set_ylabel(f'neuron (N={Cell.select().count()}) | seek')
    ax_pooled[0][1].set_ylabel(f'neuron (N={Cell.select().count()}) | hide')
else:
    ax_pooled[0][0].set_ylabel(f'neuron | seek')
    ax_pooled[0][1].set_ylabel(f'neuron | hide')

for axi in ax_pooled[1, :]:
    axi.set_xlabel('warped time (s)')
    change_ms_labels_to_sec(axi, bin_length)
    axi.set_xticks(axi.get_xticks()[::3])
    
ax_pooled[0, 2].set_ylabel('scaled rate (a.u.)')
ax_pooled[1, 2].set_visible(False)

fig.align_ylabels([ax_trial_structure_label, ax_box_closed_label])

# move the jump in label a bit to the right
last_xtick = ax2_seek.get_xticks()[-1] + (np.diff(ax2_seek.get_xticks())[-1] / 2)
new_xticks = [*ax2_seek.get_xticks()[:-1], last_xtick]
ax2_seek.set_xticks(new_xticks)
ax2_seek.set_xticklabels(hidenseek.globals.time_point_names_bo_plot[:-1])

label_subfigures([ax_trial_structure_label, ax_arena_label, ax_box_closed_label, ax_pooled_label], 'auto', -0.05);
# -

fig.savefig(os.path.join(figures_root_dir, 'intro_fig.png'), dpi = 300, bbox_inches = 'tight', pad_inches = 0)
fig.savefig(os.path.join(figures_root_dir, 'intro_fig.pdf'), dpi = 300, bbox_inches = 'tight', pad_inches = 0)

# # Cross-validation to select the number of clusters in K-means

# +
k_range = range(2, 30)

n_folds = 20

# +
pooled_rates_seek, pooled_rates_hide = make_trial_averaged_rates_for_all_sessions(bin_length, 'scale')

whole_data = xr.concat([pooled_rates_seek, pooled_rates_hide], dim = 'time')
X = whole_data.values
# -

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.mixture import GaussianMixture

# ## TimeSeriesSplit 

from sklearn.cluster import KMeans

# +
cv = TimeSeriesSplit(n_folds)

scores = []
for k in tqdm(k_range):
    for (i, (train_ind, test_ind)) in enumerate(cv.split(X)):
        km = KMeans(n_clusters = k).fit(X[train_ind, :])
        scores.append((k, i, km.score(X[test_ind, :])))
    
ts_scores_df = pd.DataFrame(scores, columns = ['k', 'fold', 'score'])

# +
fig, ax = plt.subplots(figsize = (6, 4))

sns.stripplot(x = 'k', y = 'score', data = ts_scores_df)
sns.pointplot(x = 'k', y = 'score', data = ts_scores_df)

ax.grid(True)
# -

# "Optimal" K value

df = ts_scores_df
print(df.groupby('k').score.mean().diff().argmin())
print(kneed.KneeLocator(k_range, [df.query(f"k == {k}").score.mean() for k in k_range], curve = 'concave', direction = 'increasing').knee)

# +
fig, ax = plt.subplots(figsize = (A4_width / 2, A4_height / 4))

#sns.pointplot(x = 'k', y = 'score', data = ts_scores_df, ax = ax)
ts_scores_df.groupby('k').score.mean().plot(marker = 'o')


knee_point = kneed.KneeLocator(k_range, [ts_scores_df.query(f"k == {k}").score.mean() for k in k_range], curve = 'concave', direction = 'increasing').knee
ax.axvline(x = knee_point, color = 'tab:orange')

ax.set_xlabel('number of clusters')
ax.set_ylabel('test score')

ax.set_title(f'knee = {knee_point}')

ax.grid(True)
