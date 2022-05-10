import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from hidenseek.util.postproc import *
from hidenseek.db_interface import *
from hidenseek.util.plotting import *
import hidenseek.globals

def get_stretched_y(trial, bin_length, median_time_points, z_score, smoothing_hw=500):
    """
    Put the trial's firing rates into a DataArray in which the neuron axis
    contains all the neurons from all sessions.
    Fill the rows of the neurons that were not recorded in this trial with nan
    
    Parameters
    ----------
    trial : Trial
        trial to create the spike trains for
    bin_length : float
        bin length for the spike binning in ms
    median_time_points : array-like
        reference time points for the stretching
    z_score : bool
        z-score rates or not
    
    Returns
    -------
    extended.values.T : np.array
        time x neuron numpy array
        ready to use with ssm's fit functions
        
    Example
    -------
    y = [get_y(trial, bin_length) for session in Session.select() for trial in session.trials]
    y_masked, masks = mask_input(y)
    tags = [int(session.id) for session in Session.select() for trial in session.trials]
    lls = hmm.fit(datas = y_masked, masks = masks, tags = tags)
    """
    rates = trial.get_smooth_rates(bin_length, smoothing_hw = smoothing_hw)
    stretched_rates = stretch_rates(rates, trial.time_points.values, median_time_points, bin_length).assign_coords({'neuron' : [n.id for n in trial.recorded_cells]})
    if z_score:
        stretched_rates = preproc.z_score_xr(stretched_rates)
    extended = xr.DataArray(np.nan,
                            dims = ['neuron', 'time'],
                            coords = {'neuron' : [n.id for n in Cell.select()],
                                      'time' : stretched_rates.time.values})
    
    extended.loc[stretched_rates.neuron, :] = stretched_rates
    return extended.transpose('time', 'neuron')


def make_trial_averaged_rates_for_all_sessions(bin_length, transformation):
    """
    Stretch and trial-average the firing rates of all neurons from all sessions,
    in seek and hide separately

    Parameters
    ----------
    bin_length : int
        bin length for the spike binning in ms 
    transformation : str, optional
        transformation to apply to the average rates per neuron
        if None: no transformation is done
        if "z-score": z-score the rates
        if "scale": scale the rates between 0 and 1

    Returns
    -------
    (av_rates_xr_seek, av_rates_xr_hide) : tuple of xr.DataArrays
        tuple of average rates in the two roles
    """
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)
    
    seek_y_xr = [get_stretched_y(trial, bin_length, median_time_points_seek_bo, False)
                 for session in Session.select() for trial in session.seek_trials]
    hide_y_xr = [get_stretched_y(trial, bin_length, median_time_points_hide, False)
                 for session in Session.select() for trial in session.hide_trials]
    
    rates_xr_seek = xr.concat(seek_y_xr, dim = 'trial')
    av_rates_xr_seek = rates_xr_seek.mean(dim = 'trial')
    
    rates_xr_hide = xr.concat(hide_y_xr, dim = 'trial')
    av_rates_xr_hide = rates_xr_hide.mean(dim = 'trial')
    
    if transformation is None:
        return av_rates_xr_seek, av_rates_xr_hide
    if transformation == 'z-score':
        return preproc.z_score_xr(av_rates_xr_seek), preproc.z_score_xr(av_rates_xr_hide)
    elif transformation == 'scale':
        return preproc.scale_xr(av_rates_xr_seek), preproc.scale_xr(av_rates_xr_hide)
    else:
        raise Exception('transformation has to be scale or z-score or None')


def hier_cluster_pooled_rates(rates_seek, rates_hide):
    """
    Do hierarchical clustering on the average firing rates of all neurons
    using correlation as a distance measure.

    Parameters
    ----------
    rates_seek : xr.DataArray
    rates_hide : xr.DataArray
        average firing rates returned by make_trial_averaged_rates_for_all_sessions
    
    Returns
    -------
    (idx_seek, idx_hide) : tuple of np.arrays
        sequence of indices that order neurons according
        so that neurons with similar firing rates are close together.
        For seek and hide separately.
    """
    distances_seek = pdist(rates_seek.T.values, metric = 'correlation')
    distances_hide = pdist(rates_hide.T.values, metric = 'correlation')
    Z_seek = hierarchy.ward(distances_seek)
    Z_hide = hierarchy.ward(distances_hide)

    dend_seek = hierarchy.dendrogram(Z_seek, no_plot = True)
    dend_hide = hierarchy.dendrogram(Z_hide, no_plot = True)

    idx_seek = dend_seek['leaves']
    idx_hide = dend_hide['leaves']
    
    return idx_seek, idx_hide


def kmeans_pooled_rates(rates_seek, rates_hide, k):
    """
    Apply K-means clustering in time to the trial-averaged rates of all neurons

    Parameters
    ----------
    rates_seek : xr.DataArray
    rates_hide : xr.DataArray
        average firing rates returned by make_trial_averaged_rates_for_all_sessions
    k : int
        number of clusters

    Returns
    -------
    tuple of numpy arrays with the predicted cluster labels per time point in the two roles
    """
    kmeans = KMeans(k).fit(np.concatenate([rates_seek.values, rates_hide.values]))
    return kmeans.predict(rates_seek.values), kmeans.predict(rates_hide.values)


def make_pooled_neurons_fig(ax, rates_seek, rates_hide, idx_seek, idx_hide, km_labels_seek, km_labels_hide, bin_length, k, cmap='jet', cbar=False):
    """
    Make plot about the stretched and pooled firing rates of all neurons from all sessions
    
    Parameters
    ----------
    ax : Axes
        2x2 grid
    rates_seek : xr.DataArray
        stretched and pooled rates in seek
    rates_hide : xr.DataArray
        stretched and pooled rates in hide
    idx_seek : array
        neuron indices given by hierarchical clustering
    idx_hide : array
        neuron indices given by hierarchical clustering
    bin_length : int
        binning length used for creating rates_seek and rates_hide
    k : int
        number of clusters to use for K-means on the rates
    cmap : str, default 'jet'
        colormap to use for the firing rates
    cbar : bool, default False
        draw colorbar for the rates
        if given, ax should have 3 columns
    """
    tabk, normk = get_tab20_and_norm(k)
    median_time_points_seek, median_time_points_seek_bo, median_time_points_hide = get_median_time_points_for_every_session(True)

    cbar_ax = ax[0, 2] if cbar else None

    sns.heatmap(rates_seek.T.values[idx_seek, :], cmap = cmap, ax = ax[0][0], cbar=cbar, cbar_ax=cbar_ax,
                rasterized = True)
    sns.heatmap(rates_hide.T.values[idx_hide, :], cmap = cmap, ax = ax[0][1], cbar=False,
                rasterized = True)

    sns.heatmap(km_labels_seek[None, :], cmap = tabk, norm = normk, cbar=False, ax = ax[1][0],
                rasterized = True)
    sns.heatmap(km_labels_hide[None, :], cmap = tabk, norm = normk, cbar=False, ax = ax[1][1],
                rasterized = True)

    add_vertical_lines_for_time_points(ax[0][0], median_time_points_seek_bo / bin_length, 'black', linewidth=2)
    add_vertical_lines_for_time_points(ax[0][1], median_time_points_hide / bin_length, 'black', linewidth=2)
    add_vertical_lines_for_time_points(ax[1][0], median_time_points_seek_bo / bin_length, 'black', linewidth=2)
    add_vertical_lines_for_time_points(ax[1][1], median_time_points_hide / bin_length, 'black', linewidth=2)

    for axi in ax.flatten():
        axi.set_xlabel('')
        axi.set_ylabel('')
        #axi.set_xticks([])
        axi.set_yticks([])

    for axi in ax[0, :]:
        axi.set_xticks([])

    ax2_seek = show_time_point_names(ax[0][0], median_time_points_seek_bo[:-1] / bin_length, hidenseek.globals.time_point_names_bo_plot[:-1], label_font_size=plotting_setup.font_size)
    ax2_hide = show_time_point_names(ax[0][1], median_time_points_hide[:-1] / bin_length, hidenseek.globals.time_point_names_plot[:-1], label_font_size=plotting_setup.font_size)
    
    return ax2_seek, ax2_hide

