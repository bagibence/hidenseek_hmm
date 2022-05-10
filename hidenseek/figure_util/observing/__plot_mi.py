import matplotlib.pyplot as plt
import seaborn as sns

from hidenseek.figure_util.get_p import get_p
from hidenseek.figure_util.signif_code import signif_code


def plot_mi(dfi, ax, scale=None, x_field='session_id'):
    """
    Plot real (point) and fake (violint) MI scores

    Parameters
    ----------
    dfi : pd.DataFrame
        dataframe containing the scores
    ax : matplotlib axis
        axis to plot into
    scale : bool
        scale parameter to pass to sns.pointplot
    x_field : str
        field in the dataframe that should be on the x-axis
    """
    sns.pointplot(y = 'score', x = x_field, data = dfi.query('kind == "real"'), join=None, ax = ax, color = 'tab:orange', zorder = 0, scale = scale)
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")
    
    sns.violinplot(y = 'score', x = x_field, data = dfi.query('kind == "fake"'), ax = ax, color = 'tab:blue', zorder = 100)
    
    y = dfi.score.max() * 1.1
    for (i, p) in enumerate([get_p(sdf) for (_, sdf) in dfi.groupby(x_field)]):
        ax.annotate(signif_code(p), (i, y), horizontalalignment = 'center')

    ax.set_ylim((dfi.score.min(), dfi.score.max()*1.2))
    ax.set_ylabel('MI with behvior')
    ax.set_xlabel('session ID')
