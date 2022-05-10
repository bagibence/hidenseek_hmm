import matplotlib.pyplot as plt
import seaborn as sns

from . import plotting_setup

from matplotlib.patches import Rectangle

def add_rect(ax, col, linewidth):
    """
    Add rectangle around ax

    Parameters
    ----------
    ax : Axes
        subplot around which to add the rectangle
    col : str or rgb color
        color for the rectangle
    linewidth : float
        width of the rectangle
    """
    rec = Rectangle((0, 0), 1, 1,
                    transform = ax.transAxes,
                    fill=False,
                    lw=linewidth,
                    edgecolor = col)
    rec = ax.add_patch(rec)
    rec.set_zorder(0)
    rec.set_clip_on(False)
    
    
def plot_im_on_ax(im, ax, color=None):
    """
    Draw an image on an axis

    Parameters
    ----------
    im : np.array
        image as a numpy array
    ax : Axes
        subplot to draw on
    color : str or rgb color, optional
        color of the rectangle to draw around the image
        if None, no rectangle is drawn
    """
    if color is not None:
        #with plt.rc_context({'axes.edgecolor': color}):
        #    ax.imshow(im)
        #    plt.setp(ax.spines.values(), color=color, linewidth = 5)
        ax.imshow(im)
        add_rect(ax, color, 6)
        plt.setp(ax.spines.values(), visible = False)
    else:
        ax.imshow(im)


def show_time_point_names_same_axis(ax, positions, names, labelrotation=60, label_font_size=None):
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
    label_font_size : int, optional
        font size for the labels

    Returns
    -------
    None
    """
    if label_font_size is None:
        label_font_size = plotting_setup.font_size
        
    ax.get_xaxis().set_ticks_position('top')
    ax.set_xticks(positions)
    ax.set_xticklabels(names, size = label_font_size)
    ax.tick_params(axis='x', labelrotation=labelrotation)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='left', rotation_mode='anchor');
    sns.despine(ax = ax, top = True, right = True)
    ax.tick_params(length = 0)
    ax.set_xlim(ax.get_xlim())


