import matplotlib.pyplot as plt
import seaborn as sns
import hidenseek.util.plotting.plotting_setup as plotting_setup

def show_time_point_names(ax, positions, names, labelrotation=60, label_font_size=None):
    """
    Plot time point names above the given Axes

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to write the time point names
    positions : array-like
        time point positions
    names : array-like
        time point names to write
    labelrotation : float (default 60)
        degree with which to rotate the labels

    Returns
    -------
    ax2:
        secondary x axis
    """
    if label_font_size is None:
        label_font_size = plotting_setup.font_size

    ax2 = ax.twiny()
    ax2.get_xaxis().set_visible(True)
    ax2.get_xaxis().set_ticks_position('top')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(names, size = label_font_size)
    ax2.tick_params(axis='x', labelrotation=labelrotation)
    plt.setp(ax2.xaxis.get_majorticklabels(), ha='left', rotation_mode='anchor');
    sns.despine(ax = ax2, top = True, right = True)
    ax2.tick_params(length = 0)
    ax2.set_xlim(ax.get_xlim())
    
    return ax2


def add_vertical_lines_for_time_points(ax, time_points, color, linewidth = 4):
    """
    Plot vertical lines at the time points

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    time_points : array-like
        time points / phase transitions
    color : str
        color of the vertical lines
    linewidth : int (default 4)
        width of the lines
    """
    for tp in time_points:
        ax.axvline(x = tp, color = color, linewidth = linewidth)


