import numpy as np

import string

from hidenseek.util.plotting import plotting_setup


def change_ms_labels_to_sec(ax, bin_length=1, precision=0):
    """
    Change the x axislabels from ms to second

    Parameters
    ----------
    ax : plt.Axes
        axis to relabel
    bin_length : float, default 1
        binning length of the data on the axis
    precision : int, default 0
        number of decimal places
    """
    import matplotlib.ticker as tkr
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0)
    if precision == 0:
        yfmt = tkr.FuncFormatter(lambda x, pos: f'{int(x * bin_length / 1000)}')    # create custom formatter function
    else:
        yfmt = tkr.FuncFormatter(lambda x, pos: f'{x * bin_length / 1000:.{precision}f}')    # create custom formatter function
    ax.get_xaxis().set_major_formatter(yfmt)


def label_subfigures(axes, x_offs, default_offset=-0.1):
    """
    Add ABC labels to subplots (same as label_axes, but different implementation)
    Parameters
    ----------
    axes : list of Axes objects
        subfigures to label
    x_offs : float, list of floats, 'auto'
        offset of the label
    default_offset : float
        offset for the widest subplot in axes
        used to adjust the offsets if x_offs='auto'
        
    Returns
    -------
    annotations
    """
    if x_offs == 'auto':
        x_offs = get_auto_x_offset_for_labeling(axes, default_offset)
        
    if isinstance(x_offs, float):
        x_offs = [x_offs] * len(axes)
        
    assert len(axes) == len(x_offs)
        
    annotations = []
    for axi, label, x_offset in zip(axes, string.ascii_uppercase, x_offs):
        ann = axi.set_ylabel(label, size = plotting_setup.subplot_labelsize, weight = 'bold', rotation = 0)
        annotations.append(ann)
        axi.yaxis.set_label_coords(x_offset, 1.0)

    return annotations


def get_auto_x_offset_for_labeling(axes, default_offset=-0.1):
    """
    Calculate the x offset of the ABC labels based on subplot sizes

    Parameters
    ----------
    axes : list of Axes
        subfigures to label
    default_offset : float
        x offset of the widest subfigure
        defaults to -0.1

    Returns
    -------
    x_offs : list of floats
        list of offset per axes
    """
    ax_widths = [axi.get_position().width for axi in axes]
    max_width = max(ax_widths)
    x_offs = [max_width/w * default_offset for w in ax_widths]

    return x_offs


def share_axis(subplots, axis, clear_ticks):
    """
    Share x or y axis of a set of subplots after creation
    
    Parameters
    ----------
    subplots : list of Axes
        subplots whose axis we want to share
    axis : str
        x or y
    clear_ticks : int or bool, default True
        only keep the ticks on the subplot with this index
        if True, clear all of them
        if False, keep all

    Returns
    -------
    None
    """
    for i, axi in enumerate(subplots):
        getattr(subplots[0], f'get_shared_{axis}_axes')().join(subplots[0], axi)
        if clear_ticks is True or (isinstance(clear_ticks, int) and axi is not subplots[clear_ticks]):
            getattr(axi, f'set_{axis}ticks')([])
        
    subplots[0].autoscale()


def add_identity(axes, *line_args, **line_kwargs):
    """
    Add the x=y line to axes
    """
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def unique_legend(axi, **kwargs):
    """
    Add a legend to an Axes with unique labels only.
    Sometimes legends contain the same label multiple times. This takes care of that
    
    Parameters
    ----------
    axi : matplotlib Axes
        Axes to put the legend on
    """
    handles, labels = axi.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axi.legend(by_label.values(), by_label.keys(), **kwargs)


def unique_fig_legend(fig, ax, loc=None):
    """
    Add a unique legend to the figure and remove the legends from the subplots
    """
    handles, labels = [], []

    for axi in ax.flatten():
        h, l = axi.get_legend_handles_labels()
        for hi in h:
            handles.append(hi)
        for li in l:
            labels.append(li)

    for axi in ax.flatten():
        try:
            axi.get_legend().remove()
        except:
            pass
        
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc)
