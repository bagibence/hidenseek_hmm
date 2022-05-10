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
# This figure generates Supplementary Figure 2.
#
# To generate the data to be plotted, first run the following notebooks:
# - `decode_behavioral_state`
# - `train_on_one_part_then_MI_on_other`

# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

import os
import pickle
from joblib import Parallel, delayed

import ssm
import autograd.numpy.random as npr

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures')

# %%
from hidenseek.figure_util.load_results import load_results, load_factors

K = 11
transitions = 'sticky'

_, bin_length = load_results(K, transitions)

# %%
from hidenseek.util.plotting.plotting_setup import *
from hidenseek.util.plotting import label_subfigures

# %% [markdown]
# # Load decoding results 

# %%
decoding_df = pd.read_csv(f'decode_behavior_cv_results_K_{K}_cv_10.csv')
decoding_df['paper_id'] = decoding_df.apply(lambda row: Session[row.session].paper_id, axis = 1)

# %%
import scipy.stats
from hidenseek.figure_util.signif_code import signif_code

p_tuples = []
for (sid, sdf) in decoding_df.groupby('session'):
    wilcox_stratified = scipy.stats.wilcoxon(sdf.query('method == "random_forest"').score.values,
                                             sdf.query('method == "dummy_stratified"').score.values,
                                             alternative = 'greater')
    wilcox_most_frequent = scipy.stats.wilcoxon(sdf.query('method == "random_forest"').score.values,
                                                sdf.query('method == "dummy_most_frequent"').score.values,
                                                alternative = 'greater')
    
    p_tuples.append((sid, max(wilcox_stratified.pvalue, wilcox_most_frequent.pvalue)))
    
decoding_p_df = pd.DataFrame(p_tuples, columns = ('session', 'p_value'))

# %% [markdown]
# # Load non-cross-validated MI results 

# %% tags=[]
sessionwise_scores_df = pd.read_csv(f'../real_and_fake_mi_scores_K_{K}.csv', index_col = 0)
real_mi_scores = sessionwise_scores_df.query("kind == 'real'")

real_mi_scores['num_neurons'] = [Session[sid].recorded_cells.count() for sid in real_mi_scores.session_id]

# %% [markdown]
# # Load cross-validated MI results

# %%
cross_mi_score_df = pd.read_csv(f'train_on_one_part_then_MI_on_other_K_{K}.csv')


mi_p_value = 1 / 10000
#mi_p_value = 1 / 1000


mi_df = cross_mi_score_df.query('score_type == "mi"')
adj_mi_df = cross_mi_score_df.query('score_type == "adjusted"')

p_per_fold = []
for (sid, sdf) in mi_df.groupby('session'):
    for (fold, foldf) in sdf.groupby('fold'):
        real_score = foldf.query('real == "real"').iloc[0].score.item()
        fake_scores = foldf.query('real == "fake"').score.values
        
        p_per_fold.append((sid, fold, np.mean(fake_scores > real_score)))
        
cross_mi_pdf = pd.DataFrame(p_per_fold, columns = ('session', 'fold', 'p'))


cross_mi_pdf['signif'] = cross_mi_pdf.p < mi_p_value
cross_mi_pdf['num_cells'] = [Session[sid].recorded_cells.count() for sid in cross_mi_pdf.session]

# %% [markdown]
# # Calculate MI for the circular shuffle

# %%
from hidenseek.figure_util.add_behavioral_states import add_behavioral_states
add_behavioral_states()

# %%
from hidenseek.figure_util.fake_state_functions import *

import sklearn.metrics

score_fun = sklearn.metrics.mutual_info_score
score_kwargs = {}


# %%
shift_by_whole_segments = False

for session in tqdm(Session.select()):
    session.fake_states_list = circular_shuffle_states(session, shift_by_whole_segments)

# %%
add_real_and_fake_scores_to_sessions(score_fun, score_kwargs)

by_bin_scores_df = get_scores_df()

# %%
by_bin_scores_df['paper_id'] = by_bin_scores_df.apply(lambda row: Session[row.session_id].paper_id, axis = 1)

# %%
by_bin_p_tuples = []
for (sid, subdf) in by_bin_scores_df.groupby('session_id'):
    real_score = subdf.query('kind == "real"').score.values[0]
    
    p = np.mean(subdf.query('kind == "fake"').score > real_score)
    print(f"""{p:.4f}""", end = ', ')
    
    by_bin_p_tuples.append((sid, p))
    
by_bin_pdf = pd.DataFrame(by_bin_p_tuples, columns = ('session', 'p'))


# %% [markdown]
# # Make the figure

# %%
def plot_shifted(ax, scores_df, fake_scores, real_score, paper_id, orange_dot_scale=None):
    """
    Plot the MI scores and how they depend on the shift for shifted segments
    
    Parameters
    ----------
    ax : array of Axes
        subplots to plot on
    scores_df : pd.DataFrame
        dataframe with real and fake scores for every session
    fake_scores : np.array
        fake scores as a function of shift in the example session
    real_score : float
        real score of the example session
    paper_id : int
        session ID of the example session
        (there's a difference in session.ID and session.paper_id
        because session #6 was excluded)
    orange_dot_scale : float, optional
        scale parameter to pass to sns.pointplot
        for controlling the size of the orange dots
        (showing the real scores)
        
    Returns
    -------
    None
    """
    sns.violinplot(y = 'score', x = 'paper_id', data = scores_df.query('kind == "fake"'), ax = ax[0], color = "tab:blue")
    sns.pointplot(y = 'score', x = 'paper_id', data = scores_df.query('kind == "real"'), join=None, ax = ax[0], color = "tab:orange", scale=orange_dot_scale)
    sns.despine(ax = ax[0])
    ax[0].set_xlabel("session ID")
    ax[0].set_ylabel("MI")


    ax[1].plot(fake_scores, label = "shifted", color = "tab:blue")
    ax[1].axhline(real_score, label = "real", color = "tab:orange")
    #ax[1].set_xlabel("shift (segments)")
    ax[1].set_ylabel("MI")
    ax[1].set_title(f"session {paper_id}")
    ax[1].legend(frameon = False, fancybox = False)


    sns.despine(ax = ax[1])

# %%
orange_dot_scale = 0.5
star_fontsize = 8

# %%
fig = plt.figure(figsize = (cb_width, 0.8 * cb_height), constrained_layout = False)

gs = fig.add_gridspec(nrows = 4, ncols = 1, hspace = 0.5)
gs_mi = gs[0].subgridspec(nrows = 1, ncols = 2, width_ratios = [3, 2], wspace = 0.3)
gs_shifted = gs[1].subgridspec(nrows = 1, ncols = 2, hspace = 0.1)
gs_upper = gs[2].subgridspec(nrows = 1, ncols = 2, width_ratios = [2, 1])
gs_lower = gs[3].subgridspec(nrows = 1, ncols = 2, wspace = 0.3)


ax_ref = fig.add_subplot(gs_mi[0])
ax_num = fig.add_subplot(gs_mi[1])
ax_ref_label = fig.add_subplot(gs_mi[0], xticks = [], yticks = [], frame_on = False, label = 'ref_label')
ax_num_label = fig.add_subplot(gs_mi[1], xticks = [], yticks = [], frame_on = False, label = 'num_label')

sdf = cross_mi_score_df.query('session == 13')
sns.violinplot(x = 'fold', y = 'score', hue = 'real', data = sdf.query('score_type == "mi" and real == "fake"'), label = 'fake', ax = ax_ref)
sns.pointplot(x = 'fold', y = 'score', hue = 'real', data = sdf.query('score_type == "mi" and real == "real"'),
              palette = ['tab:orange', ],
              join = False,
              label = 'real',
              scale = orange_dot_scale,
              ax = ax_ref)

y = sdf.score.max() * 1.1
for (i, p) in enumerate(cross_mi_pdf.query('session == 13').p.values):
    ax_ref.annotate(signif_code(p), (i, y), horizontalalignment = 'center', fontsize = star_fontsize)
ax_ref.set_ylim(sdf.score.min(), sdf.score.max()*1.2)

ax_ref.set_xlabel('CV fold')
ax_ref.set_ylabel('MI with behavior')
ax_ref.get_legend().remove()
sns.despine(ax = ax_ref)

ax_num.scatter([s.recorded_cells.count() for s in Session.select()],
               cross_mi_pdf.groupby('session').signif.sum().values,
               alpha = 0.5)

ax_num.set_xlabel('Number of neurons')
#ax_num.set_ylabel(f'number of p < {mi_p_value} CV folds')
ax_num.set_ylabel(f'# p < {mi_p_value} CV folds')

ax_comp = fig.add_subplot(gs_upper[0])
ax_n = fig.add_subplot(gs_lower[0])
ax_n_mi = fig.add_subplot(gs_lower[1])

ax_comp_label = fig.add_subplot(gs_upper[0], frame_on = False, xticks = [], yticks = [], label = 'rf_vs_dummy_label')
ax_n_label = fig.add_subplot(gs_lower[0], frame_on = False, xticks = [], yticks = [], label = 'num_neurons_label')
ax_n_mi_label = fig.add_subplot(gs_lower[1], frame_on = False, xticks = [], yticks = [], label = 'num_neurons_mi_label')

sns.pointplot(x = 'paper_id', y = 'score', hue = 'method',
              data = decoding_df.query('method in ["random_forest", "dummy_stratified", "dummy_most_frequent"]'),
              ax = ax_comp,
              join = None,
              #ci = None,
              scale = 0.8,
              estimator = np.mean,
              legend = False)

plt.setp(ax_comp.lines, zorder=100)
plt.setp(ax_comp.collections, zorder=100, label="")

scores_df = decoding_df.query('method in ["random_forest", "dummy_stratified", "dummy_most_frequent"]').replace({'random_forest' : 'Random forest',
                                                                                                           'dummy_stratified' : 'Dummy (stratified)',
                                                                                                           'dummy_most_frequent' : 'Dummy (most frequent)'})
sns.stripplot(x = 'paper_id', y = 'score', hue = 'method',
              data = scores_df,
              ax = ax_comp,
              alpha = 0.2)

y = scores_df.score.max() * 1.1
for (i, p) in enumerate(decoding_p_df.p_value.values):
    ax_comp.annotate(signif_code(p), (i, 0.9), horizontalalignment = 'center', fontsize = star_fontsize)
ax_comp.set_ylim((scores_df.score.min(), scores_df.score.max()*1.2))


ax_comp.legend(title = 'Method', fancybox = False, framealpha = 0, bbox_to_anchor = (1.2, 0, 0.25, 1), loc = 'center')

ax_comp.set_ylabel('Decoding accuracy')
ax_comp.set_xlabel('session ID')
sns.despine(ax = ax_comp)


    
# D)
rf_mean_df = decoding_df.query('method == "random_forest"').groupby('paper_id').mean()
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rf_mean_df.num_cells, rf_mean_df.score, alternative = 'greater')
ax_n.scatter(rf_mean_df.num_cells,
             rf_mean_df.score,
             alpha = 0.5,
             label = f'r = {r_value:.3f}\np = {p_value:.6f}')
ax_n.set_xlabel('Number of neurons')
ax_n.set_ylabel('Mean decoding accuracy')

leg = ax_n.legend(handlelength=0, handletextpad=0, fancybox=False, framealpha = 0, loc = 'lower right')
for item in leg.legendHandles:
    item.set_visible(False)


# E)
#make_reg_plot(real_mi_scores, ax_n_mi)
#make_mi_scatter_plot(real_mi_scores, ax_n_mi)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(real_mi_scores.score, rf_mean_df.score, alternative = 'greater')
ax_n_mi.scatter(real_mi_scores.score,
                rf_mean_df.score,
                alpha = 0.5,
                label = f'r = {r_value:.3f}\np = {p_value:.6f}')
ax_n_mi.set_xlabel('MI with behavior')
ax_n_mi.set_ylabel('Mean decoding accuracy')

leg = ax_n_mi.legend(handlelength=0, handletextpad=0, fancybox=False, framealpha = 0, loc = 'lower right')
for item in leg.legendHandles:
    item.set_visible(False)
    
# circular shifted MI scores
ax_by_bin = [fig.add_subplot(gs_shifted[i]) for i in range(2)]
ax_by_bin_label = [fig.add_subplot(gs_shifted[i], frame_on = False, xticks = [], yticks = [], label = f'shifted_label_{i}') for i in range(2)]
    
plot_shifted(ax_by_bin, by_bin_scores_df, session.fake_scores, session.real_score, session.paper_id, orange_dot_scale)

# add star signif code for the shifted MI scores
y = by_bin_scores_df.score.max() * 1.1
for (i, p) in enumerate(by_bin_pdf.p.values):
    ax_by_bin[0].annotate(signif_code(p), (i, y), horizontalalignment = 'center', fontsize = star_fontsize)
ax_by_bin[0].set_ylim(by_bin_scores_df.score.min(), by_bin_scores_df.score.max()*1.2)

ax_by_bin[1].set_xlabel('shift (#bins)')


x_offs = -0.175
label_subfigures([ax_ref_label, ax_num_label, *ax_by_bin_label, ax_comp_label, ax_n_label, ax_n_mi_label],# 'auto', -0.175)
                 [x_offs, x_offs, 1.2*x_offs, 0.8*x_offs, x_offs, 1.3*x_offs, 1.3*x_offs])

fig.align_ylabels([ax_ref_label, ax_by_bin_label[0], ax_comp_label, ax_n_label])

# %%
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_2.png'), dpi = 400)
fig.savefig(os.path.join(figures_root_dir, f'supp_fig_2.pdf'), dpi = 400)

# %%
