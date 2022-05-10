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
# This notebook generates Supplementary Figure 4.

# %%
import os

from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting import get_state_colors, get_tab20_and_norm

# %%
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states

K = 11
transitions = 'sticky'
n_seeds = 40

_, bin_length = load_results(K, transitions, n_seeds = n_seeds)

tab20, norm = get_tab20_and_norm(K)
state_colors = get_state_colors(K)

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')
source_fig_dir = os.path.join(figures_root_dir, 'source_images')

# %%
session = Session[13]

# %%
# load images with walls sketched out in red
im1 = plt.imread(os.path.join(source_fig_dir, 'session_13_cam1_with_walls.png'))
im2 = plt.imread(os.path.join(source_fig_dir, 'session_13_cam2_with_walls.png'))

# trace out the walls
def trace_walls(im, r_thresh=240, g_thresh=100, b_thresh=100):
    """
    Find where the walls (drawn on in red) are in the image
    
    Parameters
    ----------
    im : np.array
        rgb image as numpy array
        values should be 0-1 floats
    r_thresh : int, default, 240
        threshold for the red channel
    g_thresh : int, default 100
        threshold for green channel
    b_thresh : int, default 100
        threshold for blue channel
    
    Returns
    -------
    2D binary mask the same shape as im
    """
    red   = im[:, :, 0] * 255
    green = im[:, :, 1] * 255
    blue  = im[:, :, 2] * 255
    return (red > r_thresh) & (green < g_thresh) & (blue < b_thresh)

t1 = trace_walls(im1)
t2 = trace_walls(im2)

# extract the walls' position in xy coordinates
wall_pos1 = np.stack(np.nonzero(t1)[::-1]).T
wall_pos2 = np.stack(np.nonzero(t2)[::-1]).T

# %%
# Calculate distances to the closest wall for every time point per state 
from sklearn.metrics import pairwise_distances_argmin_min
def wall_distances(trial, filter_invalid=True):
    """
    Calculate the distance to the nearest wall in the two cameras throughout the trial
    
    Parameters
    ----------
    trial : Trial
        trial in which to calculate the distances
    filter_invalid : bool, default True
        ignore some areas in the image where the rat cannot be,
        but the tracker reports it being there
        
    Returns
    -------
    (distances1_xr, distances2_xr) : (xr.DataArray, xr.DataArray)
        distances in the two camera's images through time
    """
    p1 = trial.pos1.copy(deep=True)
    p2 = trial.pos2.copy(deep=True)
    
    if filter_invalid:
        # get rid of points that are to the left outside the arena
        p2['x'][p2.x.values < 140] = np.nan
        p2['y'][p2.x.values < 140] = np.nan
    
    rat_pos1 = p1.to_array().values.T
    rat_pos2 = p2.to_array().values.T

    valid_rat_pos1 = rat_pos1[np.isfinite(rat_pos1).all(axis=1), :]
    valid_rat_pos2 = rat_pos2[np.isfinite(rat_pos2).all(axis=1), :]

    try:
        argmins1, distances1 = pairwise_distances_argmin_min(valid_rat_pos1, wall_pos1)
    except:
        argmins1, distances1 = [], []
    try:
        argmins2, distances2 = pairwise_distances_argmin_min(valid_rat_pos2, wall_pos2)
    except:
        argmins2, distances2 = [], []

    ext_distances1 = np.full(trial.pos1.time.size, np.nan)
    ext_distances1[np.isfinite(rat_pos1).all(axis=1)] = distances1
    distances1_xr = xr.DataArray(ext_distances1, dims = ('time'), coords = {'time' : trial.pos1.time})

    ext_distances2 = np.full(trial.pos2.time.size, np.nan)
    ext_distances2[np.isfinite(rat_pos2).all(axis=1)] = distances2
    distances2_xr = xr.DataArray(ext_distances2, dims = ('time'), coords = {'time' : trial.pos2.time})
    
    return distances1_xr, distances2_xr


dist_tuples = []
for trial in tqdm(session.trials):
    distances1_xr, distances2_xr = wall_distances(trial)
    for state_id in range(K):
        state_times = trial.states.time.values[trial.states.values == state_id]

        distances_xr = xr.concat([distances1_xr.sel(time = state_times, method = 'nearest', tolerance = 250).dropna('time'),
                                  distances2_xr.sel(time = state_times, method = 'nearest', tolerance = 250).dropna('time')],
                                 dim = 'cam').min('cam')
        
        for dist in distances_xr.values:
            dist_tuples.append((state_id, trial.id, trial.role, dist))
            
            
dist_df = pd.DataFrame(dist_tuples, columns = ('state', 'trial', 'role', 'distance'))


# %%
def plot_positions(state_id, filter_invalid=True, ax=None, im_alpha=1, scatter_alpha=1):
    """
    Quickly plot the positions of the rat when a given state is active in the session
    on top of im1 and im2

    Parameters
    ----------
    state_id : int
        ID of the state to consider
    filter_invalid : bool, default True
        ignore some areas in the image where the rat cannot be,
        but the tracker reports it being there
    ax : Axes, default None
        subplot to plot on
        if None, create new figure
    im_alpha : float, default 1.
        opacity of the underlying images    
    scatter_alpha : float, default 1.
        opacity of the points showing the positions

    Returns
    -------
    if ax is None:
        (fig, ax) : (Figure, Axes)
            newly created figure and subplot
    if ax is not None:
        ax : Axes
            modified ax
    """
    state_color = tab20(norm(state_id))

    if ax is None:
        px = 1 / plt.rcParams['figure.dpi']
        plot_width = im2.shape[1]
        plot_height = im2.shape[0] + im1.shape[0]
        downscaling = 0.3
        fig, ax = plt.subplots(figsize = (plot_width*px*downscaling, plot_height*px*downscaling), nrows = 2, gridspec_kw = {'hspace' : 0}, sharex = True, constrained_layout = True)
        return_fig = True
    else:
        return_fig = False

    ax[1].imshow(im1, alpha = im_alpha)
    ax[0].imshow(im2, alpha = im_alpha)

    for trial in session.trials:
        state_times = trial.states.time.values[trial.states.values == state_id]

        p1 = trial.pos1.copy(deep=True)
        p2 = trial.pos2.copy(deep=True)

        if filter_invalid:
            # get rid of points that are to the left outside the arena
            p2['x'][p2.x.values < 140] = np.nan
            p2['y'][p2.x.values < 140] = np.nan

        p1.sel(time = state_times, method = 'nearest', tolerance = 250).dropna('time').plot.scatter('x', 'y', ax = ax[1], color = state_color, alpha = scatter_alpha)
        p2.sel(time = state_times, method = 'nearest', tolerance = 250).dropna('time').plot.scatter('x', 'y', ax = ax[0], color = state_color, alpha = scatter_alpha)
        
    ax[0].set_xlabel('')
    ax[0].set_title(f'state #{state_id}')
    
    if return_fig:
        return fig, ax
    else:
        return ax


# %%
from hidenseek.util.array_utils import reduce_dim

def add_speed(trial, filter_invalid_pos):
    """
    Approximate the speed of the animal based on the tagged positions through the trial

    Parameters
    ----------
    trial : Trial
        trial to consider
    filter_invalid_pos : bool
        ignore some areas in the image where the rat cannot be,
        but the tracker reports it being there

    Returns
    -------
    None, but adds a .speed xr.DataArray to the trial
    """
    p1 = trial.pos1.copy(deep=True)
    p2 = trial.pos2.copy(deep=True)
    if filter_invalid_pos:
        p2['x'][p2.x.values < 140] = np.nan
        p2['y'][p2.x.values < 140] = np.nan
        
    speed1 = reduce_dim(p1.to_array().differentiate('time'), np.linalg.norm, dim = 'variable')
    speed2 = reduce_dim(p2.to_array().differentiate('time'), np.linalg.norm, dim = 'variable')

    # groupby and mean to get rid of duplicate time indices
    speed = xr.concat((speed1, speed2), dim = 'time').sortby('time').groupby('time').mean()
    
    trial.speed = speed
    

def rebin_speed(trial):
    time_bins = trial.states.time.values
    dt = np.diff(time_bins)[-1]
    time_bins = np.append(time_bins, time_bins[-1] + dt)

    rebinned_speed = xr.concat([trial.speed.sel(time = slice(start, end)).mean()
                                for (start, end) in zip(time_bins[:-1], time_bins[1:])],
                               dim = 'time')
    
    trial.speed = rebinned_speed


# %% tags=[]
filter_invalid = True

for trial in tqdm(session.trials):
    add_speed(trial, filter_invalid)
    rebin_speed(trial)
    
    trial.shifted_states = trial.states.assign_coords({'time' : trial.states.time + trial.abs_time_points.start})
    trial.shifted_speed = trial.speed.assign_coords({'time' : trial.speed.time + trial.abs_time_points.start})

# %%
# set the spikes of speed (probably due to jumps in the tagged position) to nan
percent = 95

session_states = xr.concat([trial.shifted_states for trial in session.trials], 'time').sortby('time')
session_speed = xr.concat([trial.shifted_speed for trial in session.trials], 'time').sortby('time')

cutoff = np.percentile(session_speed[~np.isnan(session_speed)], percent)
session_speed[session_speed > cutoff] = np.nan

# %%
speed_tuples = []
for i in range(K):
    for s in session_speed[session_states.values == i].values:
        speed_tuples.append((i, s))
        
speed_df = pd.DataFrame(speed_tuples, columns = ['state', 'speed'])

# %%
from hidenseek.util.plotting import label_subfigures

def plot_measurements_horizontal(plot_df, var_name, ax, xlabel, kind, estimator):
    """
    Plot speed or distance from the wall per state

    Parameters
    ----------
    plot_df : pd.DataFrame
        dataframe containing speeds / distances
    var_name : string
        name of the variable to plot
        'distance' or 'speed'
    ax : Axes
        subplot to plot into
    xlabel : str
        xlabel
    kind : str
        kind of plot to make
        'swarm' or 'strip'
    estimator : function
        aggregating function for the pointplot
        np.mean and np.median

    Returns
    -------
    None
    """
    if kind == 'swarm':
        sns.swarmplot(y = 'state', x = var_name, data = plot_df, ax = ax, hue = 'state', palette = [tab20(norm(i)) for i in range(K)], s=1, zorder = 0, orient = 'horizontal')
    elif kind == 'strip':
        sns.stripplot(y = 'state', x = var_name, data = plot_df, ax = ax, alpha = 0.2, zorder = 0, hue = 'state', palette = [tab20(norm(i)) for i in range(K)], orient = 'horizontal')
    else:
        raise Exception('kind has to be strip or swarm')
        
    if estimator == 'mean':
        estimator = np.mean
    if estimator == 'median':
        estimator = np.median
    sns.pointplot(y = 'state', x = var_name, data = plot_df, estimator = estimator, color = 'black', ax = ax, ci = None, join = False, zorder = 1, orient = 'horizontal')

    ax.get_legend().remove()
    ax.set_ylabel('HMM state')
    ax.set_xlabel(xlabel)


# %%
im_alpha = 1
scatter_alpha = 1
estimator_speed = 'median'
estimator_wall = 'median'

wall_states = [4, 9]
darting_state = 7
transit_state = 5


# %%
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height))

gs = fig.add_gridspec(nrows = 2, ncols = 2,
                      height_ratios = [1, 1],
                      width_ratios = [1, 1])
gs_wall_state = gs[1, 1].subgridspec(nrows = 2, ncols = 1, hspace = 0)
gs_fast_states = gs[0, 1].subgridspec(nrows = 2, ncols = 1, hspace = 0)

ax_dist_per_state = fig.add_subplot(gs[1, 0])
ax_wall_state = [fig.add_subplot(gsi) for gsi in gs_wall_state]

ax_speed_per_state = fig.add_subplot(gs[0, 0])
ax_fast_states = [fig.add_subplot(gsi) for gsi in gs_fast_states]


for i in wall_states:
    plot_positions(i, ax = ax_wall_state, im_alpha = im_alpha, scatter_alpha = scatter_alpha)
ax_wall_state[0].set_title('"Wall"')

plot_positions(darting_state, ax = ax_fast_states, im_alpha = im_alpha, scatter_alpha = scatter_alpha)
plot_positions(transit_state, ax = ax_fast_states, im_alpha = im_alpha, scatter_alpha = scatter_alpha)
ax_fast_states[0].set_title('States with highest speed')

sns.despine(ax = ax_dist_per_state)
sns.despine(ax = ax_speed_per_state)

for axi in [*ax_wall_state, *ax_fast_states]:
    axi.set_xticks([])
    axi.set_yticks([])
    axi.set_xticklabels([])
    axi.set_yticklabels([])
    axi.set_xlabel('')
    axi.set_ylabel('')
    
plot_measurements_horizontal(dist_df, 'distance', ax_dist_per_state, 'Distance from wall (a.u.)', 'strip', estimator_wall)
plot_measurements_horizontal(speed_df, 'speed', ax_speed_per_state, 'Speed (a.u.)', 'strip', estimator_speed)

ax_dist_per_state_label = fig.add_subplot(gs[1, 0], frame_on = False, xticks = [], yticks = [], label = 'dist_per_state_label')
ax_speed_per_state_label = fig.add_subplot(gs[0, 0], frame_on = False, xticks = [], yticks = [], label = 'speed_per_state_label')

ax_wall_state_label = fig.add_subplot(gs[1, 1], frame_on = False, xticks = [], yticks = [], label = 'wall_state_label')
ax_fast_states_label = fig.add_subplot(gs[0, 1], frame_on = False, xticks = [], yticks = [], label = 'fast_states_label')

label_subfigures([ax_speed_per_state_label, ax_fast_states_label, ax_dist_per_state_label, ax_wall_state_label], [-0.1, -0.03, -0.1, -0.03]);


# %%
fig.savefig(os.path.join(figures_root_dir, f"fig_wall_and_darting_quantification_K_{K}_{n_seeds}_seeds.png"), dpi = 400)
fig.savefig(os.path.join(figures_root_dir, f"fig_wall_and_darting_quantification_K_{K}_{n_seeds}_seeds.pdf"), dpi = 400)

# %%
