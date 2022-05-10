import matplotlib.pyplot as plt
import seaborn as sns

def __pick_hmm_labels(ax, orient):
    if orient == 'horizontal':
        return ax.xaxis.get_ticklabels()
    elif orient == 'vertical':
        return ax.yaxis.get_ticklabels()
    else:
        raise Exception('orient has to be either horizontal or vertical')


def make_fig_hmm_behavior_conditional_probs(cont_df_hmm, cont_df_beh, ax1, ax2, ax_cbar, state_colors, signif_cont_df_hmm=None, signif_cont_df_beh=None, cbar_orient='vertical', remove_top_xticks=True):
    """
    Plot P(HMM state | behavioral state) and P(behavioral state | HMM state) conditional probability matrices

    Parameters
    ----------
    cont_df_hmm : pd.DataFrame
        P(behavior | HMM state)
    cont_df_beh : pd.DataFrame
        P(HMM state | behavior)
    ax1 : matplotlib Axes
        axes for P(behavior | HMM state)
    ax2 : matplotlib Axes
        axes for P(HMM state | behavior)
    ax_cbar : matplotlib Axes
        axis to put the colorbar on
    state_colors : list
        list of colors for the states
        used for coloring the numbers
    signif_cont_df_hmm : pd.DataFrame
        significancy mask for P(behavior | HMM state)
    signif_cont_df_beh : pd.DataFrame
        significancy mask for P(HMM state | behavior)
    cbar_orient : str, default 'vertical'
        orientation of the colorbar
    remove_top_xticks : bool, default True
        remove the xticks of the subplot that is on the top
    """
    k_vals_hmm = cont_df_hmm.index.values
    k_vals_beh = cont_df_beh.index.values

    if signif_cont_df_hmm is not None:
        #sns.heatmap(cont_df_hmm, mask = ~signif_cont_df_hmm, annot = True, ax = ax1, cbar = False, vmin = 0, vmax = 1)
        sns.heatmap(cont_df_hmm, mask = ~(signif_cont_df_hmm.astype(bool) & (cont_df_hmm > 0.001)), annot = True, ax = ax1, cbar = False, vmin = 0, vmax = 1,
                    rasterized = True)
        for text in ax1.texts:
            text.set_text(r'$\ast$')
    if signif_cont_df_beh is not None:
        #sns.heatmap(cont_df_beh, mask = ~signif_cont_df_beh, annot = True, ax = ax2, cbar = False, vmin = 0, vmax = 1)
        sns.heatmap(cont_df_beh, mask = ~(signif_cont_df_beh.astype(bool) & (cont_df_beh > 0.001)), annot = True, ax = ax2, cbar = False, vmin = 0, vmax = 1,
                    rasterized = True)
        for text in ax2.texts:
            text.set_text(r'$\ast$')

    sns.heatmap(cont_df_hmm, ax = ax1, vmin = 0, vmax = 1, cbar_ax = ax_cbar, cbar_kws={"orientation": cbar_orient}, yticklabels = k_vals_hmm,
                rasterized = True)
    sns.heatmap(cont_df_beh, ax = ax2, vmin = 0, vmax = 1, cbar = False, yticklabels = k_vals_beh,
                rasterized = True)


    for axi in [ax1, ax2]:
        hmm_labels = __pick_hmm_labels(axi, 'vertical')
        for t in hmm_labels:
            t.set_color(state_colors[int(t.get_text())])
            t.set_weight('bold')
            
    ax1.set_title('P(behavior | HMM state)')
    ax2.set_title('P(HMM state | behavior)')

    ax1.set_xlabel('')

    if remove_top_xticks:
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_xticks([])

